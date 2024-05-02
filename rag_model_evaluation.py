# Databricks notebook source
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.30.2 unstructured[pdf,docx]==0.10.30 llama-index==0.9.3 mlflow==2.10.1
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run /Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC #Creating OpenAI ChatGPT 3.5 as the judge for evaluation by setting up an endpoint
# MAGIC ---> Since we cant create an endpoint now, it defaults to llama2-70-B as the judge

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

try:
    endpoint_name  = "dbdemos-azure-openai"
    deploy_client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "name": endpoint_name,
                    "external_model": {
                        "name": "gpt-35-turbo",
                        "provider": "openai",
                        "task": "llm/v1/chat",
                        "openai_config": {
                            "openai_api_type": "azure",
                            "openai_api_key": "{{secrets/dbdemos/azure-openai}}", #Replace with your own azure open ai key
                            "openai_deployment_name": "dbdemo-gpt35",
                            "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
                            "openai_api_version": "2023-05-15"
                        }
                    }
                }
            ]
        }
    )
except Exception as e:
    if 'RESOURCE_ALREADY_EXISTS' in str(e):
        print('Endpoint already exists')
    else:
        print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
        endpoint_name = "databricks-llama-2-70b-chat"

#Let's query our external model endpoint
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}]})
answer_test['choices'][0]['message']['content']

# COMMAND ----------

# MAGIC %md
# MAGIC #RAF model evaluation - offline

# COMMAND ----------

volume_folder =  f"/Volumes/main/rag_chatbot/volume_databricks_documentation/evaluation_dataset"
# #Load the eval dataset from the repository to our volume
# upload_dataset_to_volume(volume_folder)
volume_folder

# COMMAND ----------


