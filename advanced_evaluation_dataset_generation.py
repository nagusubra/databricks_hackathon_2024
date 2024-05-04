# Databricks notebook source
# MAGIC %md
# MAGIC #Install libraries and modules

# COMMAND ----------

# MAGIC %pip install -q mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.30.2 unstructured[pdf,docx]==0.10.30 llama-index==0.9.3 mlflow==2.10.1
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Import libraries and modules

# COMMAND ----------

# MAGIC %run /Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

from unstructured.partition.auto import partition
import re
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from mlflow.deployments import get_deploy_client
import io
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableBranch
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough
import json


from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, TimestampType, StringType
from pyspark.sql.functions import udf
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
from pyspark.sql.functions import col, when
from typing import Iterator
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC #Import constants

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{db}/volume_oem_documentation"
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

# COMMAND ----------

# MAGIC %md
# MAGIC #Create OEM documentation volume storage

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS volume_oem_documentation;

# COMMAND ----------

# MAGIC %md
# MAGIC #PDF chunking with sentance splitter

# COMMAND ----------

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections])


# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_small_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=350, chunk_overlap=10)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ###LLM chat model and embedding model selection

# COMMAND ----------

chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 200) # 1/4 the cost of DBRX, but accuracy and performance not better than DBRX and matches the performance of llama 3
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Connecting to vector store

# COMMAND ----------

index_name=f"{catalog}.{db}.pdf_transformed_self_managed_vector_search_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope="dbdemos", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en", managed_embeddings = False)

# COMMAND ----------

print(VECTOR_SEARCH_ENDPOINT_NAME)
print(index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Langchains for question and answers 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Retriever model function to connect to the vector store and vector search endpoint created for this project

# COMMAND ----------

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Prompt templates

# COMMAND ----------

bare_prompt_template = """
                            {content}
                       """

bare_template = PromptTemplate(
    input_variables = ["content"],
    template=bare_prompt_template
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Question

# COMMAND ----------

qa_template = """
You are a University Professor creating a test for advanced students in solar energy. For each given context, create one descriptive question that is specific to the given context. Avoid creating generic or general questions. Avoid out of context questions. Avoid creating more than one question.

This is the context for you: {context}

Format the output as JSON with the following key:

"question": string  // provide a question about the context.
"""

qa_question_prompt_template = PromptTemplate(
    input_variables = ["context"],
    template=qa_template,
)

# COMMAND ----------

# # testing the question generation chain

# messages = qa_question_prompt_template.format(
#     context=["context"],
#     format_instructions='''
#                             The output should be a markdown code snippet formatted in the following schema:
#                             {"question": string  // a question about the context.}
#                         '''
# )

# question_generation_chain = (bare_template | chat_model |  StrOutputParser())
# response = question_generation_chain.invoke({"content" : messages})
# response

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Answer

# COMMAND ----------

qa_answer_template = """
You are a University Professor creating a test for advanced students in solar energy. For each question and context, create one descriptive answer. Avoid creating generic or general answers. Avoid out of context answers. Avoid creating more than one answer.

This is the context for you: {context}
This is the question for you: {question}

Format the output as JSON with the following key:

"answer": string  // provide an answer to the question using the given context.
"""

qa_answer_prompt_template = PromptTemplate(
    input_variables = ["question", "context"],
    template=qa_answer_template,
)

# COMMAND ----------

# # testing answer generation chain

# messages = qa_answer_prompt_template.format(
#     context=["context"],
#     question=''' \n{\n"question": "In what situations might a \'Device restart\' be recommended, as suggested in the provided context?"\n} ''',
#     format_instructions='''
#                             The output should be a markdown code snippet formatted in the following schema:
#                             { "answer": string  // an answer to the question}
#                         '''
# )

# answer_generation_chain = (bare_template | chat_model |  StrOutputParser())
# response = answer_generation_chain.invoke({"content" : messages})
# response

# COMMAND ----------

# MAGIC %md
# MAGIC #### Question and Answer function

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
@pandas_udf(StringType())
def generate_questions_column(contents: pd.Series) -> pd.Series:
    '''
    Generates a series of questions based on the given contents.

    Args:
        contents (pd.Series): The content input to generate questions from.

    Returns:
        pd.Series: A series of generated questions.

    Raises:
        None

    Notes:
        - This function processes the contents in batches.
        - Each batch is split into chunks to match the model's input size requirement.
        - Questions are generated based on the context provided in each batch.
        - The output is a series of generated questions in the same order as the input.

    '''

    # Process each batch and collect the results

    # # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    # max_batch_size = 1
    # batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in contents:
        batch = [str(i) for i in batch]
        messages = qa_question_prompt_template.format(
                                                context = batch,
                                                format_instructions =   '''
                                                                        The output should only have one question in a markdown code snippet formatted in the following schema:

                                                                        {"question": string  // a question about the context.}
                                                                        '''
                                                )

        question_generation_chain = (bare_template | chat_model | StrOutputParser())
        response = question_generation_chain.invoke({"content" : messages})
        try:
            question = json.loads(response)["question"]
        except:
            question = response

        if question == None:
            question = "No question, skip this one"

        all_embeddings += [question]

    return pd.Series(all_embeddings)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
@pandas_udf(StringType())
def generate_answers_column(contents: pd.Series, questions: pd.Series) -> pd.Series:
    '''
    Generates a series of answers based on the given contents and corresponding questions.

    Args:
        contents (pd.Series): The content input to generate answers from.
        questions (pd.Series): The questions to generate answers for, corresponding to the contents.

    Returns:
        pd.Series: A series of generated answers.

    Raises:
        None

    Notes:
        - This function processes the contents and questions together.
        - It creates a dictionary for each content-question pair.
        - The output is a series of generated answers in the same order as the input.
    '''

    # Process each batch and collect the results

    # # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    # max_batch_size = 1
    # batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results

    dictionary_list = []

    for index in range(len(contents)):
        content = contents[index]
        question = questions[index]

        dictionary = {
            "content": content,
            "question": question
        }

        dictionary_list.append(dictionary)

    all_embeddings = []
    for batch in dictionary_list:
        messages = qa_answer_prompt_template.format(
                                                context = batch["content"],
                                                question = batch["question"],
                                                format_instructions =   '''
                                                                        The output should only have one answer in a markdown code snippet formatted in the following schema, only one answer allowed:
                                                                        {"answer": string  // an answer to the question.}
                                                                        '''
                                                )

        answer_generation_chain = (bare_template | chat_model | StrOutputParser())
        response = answer_generation_chain.invoke({"content" : messages})
        try:
            answer = json.loads(response)["answer"]
        except:
            answer = response

        if answer == None:
            answer = "No answer, skip this one"

        all_embeddings += [answer]

    return pd.Series(all_embeddings)

# COMMAND ----------

# # important tester function
# df = spark.sql("SELECT id, url, content FROM main.asset_nav.pdf_evaluation Limit 10")
# df = df.withColumn("question", generate_questions_column("content"))
# df = df.withColumn("answer", generate_answers_column("content", "question"))

# display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Create pdf_pre_evaluation

# COMMAND ----------

# %sql
# DROP TABLE main.asset_nav.pdf_pre_evaluation

# COMMAND ----------

# dbutils.fs.rm('dbfs:/Volumes/main/asset_nav/volume_oem_documentation/checkpoints/pdf_pre_eval_chunk', True)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS main.asset_nav.pdf_pre_evaluation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   question STRING,
# MAGIC   answer STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# (spark.readStream.table('pdf_raw')
#       .withColumn("content", F.explode(read_as_small_chunk("content")))
#       .selectExpr('path as url', 'content')
#   .writeStream
#     .trigger(availableNow=True)
#     .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_pre_eval_chunk')
#     .table('pdf_pre_evaluation').awaitTermination())

# COMMAND ----------

# MAGIC %sql SELECT * FROM pdf_pre_evaluation LIMIT 2

# COMMAND ----------

# MAGIC %md
# MAGIC #Create pdf_evaluation

# COMMAND ----------

# %sql
# DROP TABLE main.asset_nav.pdf_evaluation

# COMMAND ----------

# dbutils.fs.rm('dbfs:/Volumes/main/asset_nav/volume_oem_documentation/checkpoints/pdf_eval_chunk', True)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS main.asset_nav.pdf_evaluation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   question STRING,
# MAGIC   answer STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC #Populate pdf_evaluation by appending 5 rows at a time

# COMMAND ----------

# import math

# # Get the total count of rows in the table
# row_count = spark.sql("SELECT COUNT(*) FROM main.asset_nav.pdf_pre_evaluation").collect()[0][0]

# batch_size = 5
# batch_jump_size = 5
# iterations = math.ceil((row_count)/batch_size)


# # Loop through the table in batches of 10 rows
# for i in range(iterations): # last successful i = 158, once running 158, remove it for normal range operation

#     # Fetch 10 rows starting from the current iteration
#     offset = i * batch_jump_size

#     print("Iteration #", i, " and processing row number", offset)

#     query = f"SELECT url, content FROM main.asset_nav.pdf_pre_evaluation LIMIT {batch_size} OFFSET {offset}"
#     df = spark.sql(query)
    
#     # Apply transformations

#     try:
#         df = df.withColumn("question", generate_questions_column(df["content"]))
#         df = df.withColumn("answer", generate_answers_column(df["content"], df["question"]))
#     except:
#         pass

#     # Display the transformed DataFrame
#     # display(df)

#     try:
#         df.write.format("delta").mode("append").saveAsTable("main.asset_nav.pdf_evaluation")
#     except:
#         print("Iteration #", i, " and processing row number", offset, "    FAILED to convert to delta table !!! So skipping this one.")

# COMMAND ----------

# MAGIC %md
# MAGIC #Populate pdf_evaluation table

# COMMAND ----------

# (spark.readStream.table('pdf_raw')
#       .withColumn("content", F.explode(read_as_small_chunk("content")))
#       .withColumn("question", generate_questions_column("content"))
#       .withColumn("answer", generate_answers_column("content", "question"))
#       .selectExpr('path as url', 'content', 'question', 'answer')
#       .limit(21)  # Select only the top 10 records
#   .writeStream
#     .trigger(availableNow=True)
#     .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_eval_chunk')
#     .table('pdf_evaluation').awaitTermination())

# COMMAND ----------

# MAGIC %sql SELECT * FROM pdf_evaluation LIMIT 2
