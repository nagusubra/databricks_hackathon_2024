# Databricks notebook source
# MAGIC %pip install -q mlflow[databricks]==2.10.1 lxml==4.9.3 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.30.2 unstructured[pdf,docx]==0.10.30 llama-index==0.9.3 langchain openai ragas arxiv pymupdf chromadb wandb tiktoken datasets mlflow==2.10.1 datasets tqdm
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run /Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluation using RAGAS

# COMMAND ----------

# MAGIC %pip install -U -q langchain openai ragas arxiv pymupdf chromadb wandb tiktoken datasets mlflow==2.10.1 datasets tqdm databricks-vectorsearch==0.22 transformers torch langchain-together

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate
from datasets import Dataset
from tqdm import tqdm

# COMMAND ----------

# MAGIC %run "/Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/config"

# COMMAND ----------

from mlflow import MlflowClient


# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import mlflow
import os
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
model_name = f"{catalog}.{db}.asset_nav_chatbot_model_version_1"
model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

# COMMAND ----------



# COMMAND ----------

# from langchain_together import Together

# llm = Together(
#     model="togethercomputer/RedPajama-INCITE-7B-Base",
#     temperature=0.7,
#     max_tokens=128,
#     top_k=1,
#     together_api_key="ffc9a1325fe801c3244b69aedc57cbb8fd9969330b256a42ab9bf51678e5fd0e"
# )

# input_ = """You are a teacher with a deep knowledge of machine learning and AI. \
# You provide succinct and accurate answers. Answer the following question: 

# What is a large language model?"""
# print(llm.invoke(input_))

# COMMAND ----------

from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings

together_key = "ffc9a1325fe801c3244b69aedc57cbb8fd9969330b256a42ab9bf51678e5fd0e"

together_embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# together_completion = Together(
#     model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
#     temperature=0.7,
#     max_tokens=4000,
#     top_k=1,
#     together_api_key=together_key
# )

together_completion = Together(
    model="togethercomputer/RedPajama-INCITE-7B-Base",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    together_api_key="ffc9a1325fe801c3244b69aedc57cbb8fd9969330b256a42ab9bf51678e5fd0e"
)

input_ = """You are a teacher with a deep knowledge of machine learning and AI. \
You provide succinct and accurate answers. Answer the following question: 

What is a large language model?"""
print(together_completion.invoke(input_))

# COMMAND ----------

# dialog = {
#     "messages": [
#         {"role": "user", "content": "What is an Inverter?"}, 
#         {"role": "assistant", "content": "A power inverter, or invertor is a power electronic device or circuitry that changes direct current to alternating current. The resulting AC frequency obtained depends on the particular device employed."}, 
#         {"role": "user", "content": "What are all the fault codes in KACO inverter for overheating because of fans? and give me the ways to solve it."}
#     ]
# }
# print(f'Testing with relevant history and question...')
# response = rag_model.invoke(dialog)
# display_chat(dialog["messages"], response)

# COMMAND ----------

# response["result"]

# COMMAND ----------

def create_ragas_dataset(rag_model, eval_dataset):
  rag_dataset = []
  for index, row in tqdm(eval_dataset.iterrows()):
    answer = rag_model.invoke({"messages": [{"role": "user", "content": row["question"]}]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["result"],
         "contexts" : [row["context"]],
         "ground_truth" : row["ground_truth"]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset, llm_model=together_completion, embeddings_model=together_embeddings):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
    llm=llm_model, # together_completion
    embeddings=embeddings_model # embeddings
  )
  return result



# from langchain_together import Together
# from langchain_together.embeddings import TogetherEmbeddings

# together_key = "<your-key-here>"

# embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# together_completion = Together(
#     model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
#     temperature=0.7,
#     max_tokens=4000,
#     top_k=1,
#     together_api_key=together_key
# )

# COMMAND ----------

from tqdm import tqdm
import pandas as pd
from datasets import Dataset

eval_dataset = pd.read_csv("/Volumes/main/asset_nav/volume_oem_documentation/evaluation_data/solar_rag_pipeline_evaluation_datatset_manual_with_context.csv").drop(columns="id")
display(eval_dataset)

basic_qa_ragas_dataset = create_ragas_dataset(rag_model, eval_dataset)

# COMMAND ----------

display(basic_qa_ragas_dataset)

# COMMAND ----------

basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)

# COMMAND ----------



# COMMAND ----------

# import mlflow.langchain
# model_version_uri = "models:/databricks_dbrx_models.models.dbrx_instruct/1"
# model_version_uri = "models:/databricks_dbrx_models.models.dbrx_base/1"
# model_version_uri = "models:/databricks_meta_llama_3_models.models.meta_llama_3_8b/1"
# model_version_uri = "models:/databricks_mistral_models.models.mistral_7b_v0_1/1"

# llm_model = mlflow.langchain.load_model(model_version_uri)

# COMMAND ----------

# import mlflow
# import pandas as pd

# mlflow.set_registry_uri("databricks-uc")

# logged_model = f"models:/databricks_meta_llama_3_models.models.meta_llama_3_8b/1"
# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)

# loaded_model._model_impl.pipeline("What is ML?")

# COMMAND ----------

# # code from ragas chatbot in the website

# from langchain.chat_models import AzureChatOpenAI
# from ragas.llms import LangchainLLM
# from ragas.metrics import faithfulness
# from ragas import evaluate

# # Initialize the AzureChatOpenAI model
# azure_model = AzureChatOpenAI(
#     deployment_name="your-deployment-name",
#     model="your-model-name",
#     openai_api_base="https://your-endpoint.openai.azure.com/",
#     openai_api_type="azure",
# )

# # Wrap the azure_model with LangchainLLM
# ragas_azure_model = LangchainLLM(azure_model)

# # Use the new model in the evaluate function
# result = evaluate(
#     your_dataset["eval"].select(range(10)),  # replace with your dataset
#     metrics=[faithfulness],
#     llm=ragas_azure_model,
# )

# print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC #Model is PRD ready

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)
