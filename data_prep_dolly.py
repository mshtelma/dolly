# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation of the  Dolly Dataset
# COMMAND ----------
import pandas as pd
from pyspark.sql import SparkSession, Window
import pandas as pd
from pyspark.sql.connect.functions import rand, row_number

if globals().get("spark") is None:
    spark = SparkSession.builder.master("local[1]").getOrCreate()

from datasets import load_dataset

from training.consts import (
    DEFAULT_TRAINING_DATASET,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
)

hf_ds = load_dataset(DEFAULT_TRAINING_DATASET)["train"]

spark_df = spark.createDataFrame(hf_ds.to_pandas())


def generate_instructions_udf(it):
    for pdf in it:
        recs_with_ctx_df = pdf[~pdf["context"].isnull()]
        recs_without_ctx_df = pdf[pdf["context"].isnull()]

        recs_with_ctx_df["text"] = recs_with_ctx_df.apply(
            lambda rec: PROMPT_WITH_INPUT_FORMAT.format(
                instruction=rec["instruction"],
                response=rec["response"],
                input=rec.get("context"),
            ),
            axis=1,
        )
        recs_without_ctx_df["text"] = recs_with_ctx_df.apply(
            lambda rec: PROMPT_NO_INPUT_FORMAT.format(
                instruction=rec["instruction"], response=rec["response"]
            ),
            axis=1,
        )

        yield pd.concat([recs_with_ctx_df[["text"]], recs_without_ctx_df[["text"]]])


def store_as_delta(df, name):
    w = Window().orderBy(rand())
    df.withColumn("id", row_number().over(w)).write.format("delta").mode(
        "overwrite"
    ).save(name)


instr_df = spark_df.mapInPandas(generate_instructions_udf, schema="text string")
display(instr_df)

# COMMAND ----------
res = instr_df.randomSplit([0.9, 0.1])
instr_train_df = res[0]
instr_test_df = res[1]
# COMMAND ----------

store_as_delta(instr_train_df, "/tmp/msh/datasets/dolly/dolly_instr_train.delta")
store_as_delta(instr_test_df, "/tmp/msh/datasets/dolly/dolly_instr_test.delta")
