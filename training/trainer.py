# Copyright 2023 Databricks, Inc.
import json

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click
import deepspeed
import numpy as np
from datasets import Dataset, load_dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deltatorch import create_pytorch_dataloader, FieldSpec
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
    WEIGHTS_NAME,
)
import torch
from transformers.deepspeed import is_deepspeed_zero3_enabled

from training.consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
    DEFAULT_TRAINING_DATASET,
)

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                original_text = self.tokenizer.decode(batch["labels"][i])
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}\n\n {original_text}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def load_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_batch(
    item: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int
) -> dict:
    tokenized_result = tokenizer(
        item["text"],
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    item = {}
    item["input_ids"] = tokenized_result["input_ids"][0][:max_length]
    item["attention_mask"] = tokenized_result["attention_mask"][0][:max_length]

    return item


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    dataset_path: str,
    batch_size: int = 8,
    collate_fn=None,
) -> DataLoader:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )

    return create_pytorch_dataloader(
        dataset_path,
        id_field="id",
        fields=[
            FieldSpec("text", full_record_transform=_preprocessing_function),
        ],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def get_optimizer_grouped_parameters(
    model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight"]
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def save_hf_format(model, tokenizer, output_dir="/local_disk0/model"):
    model_to_save = model.module if hasattr(model, "module") else model
    tokenizer.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    if is_deepspeed_zero3_enabled():
        file = os.path.join(output_dir, WEIGHTS_NAME)
        if os.path.isfile(file):
            os.remove(file)

        if not model.save_16bit_model(output_dir, WEIGHTS_NAME):
            logger.warning(
                "deepspeed.save_16bit_model didn't save the model, since"
                " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                " zero_to_fp32.py to recover weights"
            )
            model.save_checkpoint(output_dir)


def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    train_path: str,
    test_path: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    offload: bool,
    deepspeed_conf: str,
    gradient_checkpointing: bool,
    local_rank: str,
    warmup_steps: int,
):
    set_seed(seed)
    local_rank = int(local_rank)
    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model,
        gradient_checkpointing=gradient_checkpointing,
    )

    gradient_accumulation_steps = 1
    with open(deepspeed_conf) as json_data:
        ds_config = json.load(json_data)
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["train_batch_size"] = (
        per_device_train_batch_size
        * torch.distributed.get_world_size()
        * gradient_accumulation_steps
    )

    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    train_dataloader = preprocess_dataset(
        tokenizer,
        max_length,
        train_path,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
    )

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, 1)

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name="cosine",  # lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args={"local_rank": local_rank, "deepspeed_config": ds_config},
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Instantiating Trainer")

    logger.info("Training")
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()

    if dbfs_output_dir:
        logger.info(f"Saving Model to DBFS: {dbfs_output_dir}")
        save_hf_format(model, tokenizer, dbfs_output_dir)

    # if local_output_dir:
    #     logger.info(f"Saving Model locally: {local_output_dir}")
    #     save_hf_format(model, tokenizer, local_output_dir)

    logger.info("Done.")


@click.command()
@click.option(
    "--input-model",
    type=str,
    help="Input model to fine tune",
    default=DEFAULT_INPUT_MODEL,
)
@click.option(
    "--train-path",
    type=str,
    help="Path to the Delta Table used for training",
    required=True,
)
@click.option(
    "--test-path",
    type=str,
    help="Path to the Delta Table used for training",
    required=True,
)
@click.option(
    "--local-output-dir",
    type=str,
    help="Write directly to this local path",
    required=True,
)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option(
    "--per-device-train-batch-size",
    type=int,
    default=8,
    help="Batch size to use for training.",
)
@click.option(
    "--per-device-eval-batch-size",
    type=int,
    default=8,
    help="Batch size to use for evaluation.",
)
@click.option(
    "--test-size",
    type=int,
    default=1000,
    help="Number of test records for evaluation, or ratio of test records.",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=None,
    help="Number of steps to warm up to learning rate",
)
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option(
    "--eval-steps",
    type=int,
    default=50,
    help="How often to run evaluation on test records",
)
@click.option(
    "--save-steps", type=int, default=400, help="How often to checkpoint the model"
)
@click.option(
    "--save-total-limit",
    type=int,
    default=10,
    help="Maximum number of checkpoints to keep on disk",
)
@click.option(
    "--lr", type=float, default=1e-5, help="Learning rate to use for training."
)
@click.option(
    "--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training."
)
@click.option(
    "--deepspeed_conf", type=str, default=None, help="Path to deepspeed config file."
)
@click.option("--offload", type=bool, default=False, help="Offload")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=int,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option(
    "--bf16", type=bool, default=None, help="Whether to use bf16 (preferred on A100's)."
)
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
