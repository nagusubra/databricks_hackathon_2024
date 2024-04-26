# Databricks notebook source
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Import libraries and modules

# COMMAND ----------

# MAGIC %run /Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC #Important functions

# COMMAND ----------

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ##LLM chat model and embedding model selection

# COMMAND ----------

# # we could try LLama 2 or LLama 3 or Mistral 7B -> least expensive or Mistrall 30B or dbrx -> most expensive

# chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200) # most expensive, but faster results and more accurate than Llaama 3

# chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens = 200) # half the cost of DBRX, but accuracy and performance not better than DBRX

# chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200) # 1/4 the cost of DBRX, but accuracy and performance not better than DBRX (bigger miss in accuracy)

chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 200) # 1/4 the cost of DBRX, but accuracy and performance not better than DBRX and matches the performance of llama 3

# chat_model = ChatDatabricks(endpoint="databricks-mpt-7b-instruct", max_tokens = 200) # 404 error

# chat_model = ChatDatabricks(endpoint="databricks-mpt-30b-instruct", max_tokens = 200) # 404 error

# COMMAND ----------

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Connecting to vector store

# COMMAND ----------

index_name=f"{catalog}.{db}.pdf_transformed_self_managed_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope="dbdemos", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en", managed_embeddings = False)

# COMMAND ----------

print(VECTOR_SEARCH_ENDPOINT_NAME)
print(index_name)

# COMMAND ----------

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Validating input query chat - chain

# COMMAND ----------

validate_if_input_question_is_relevant_to_solar_energy = """
You are classifying documents to know if this question is related to Solar Energy, Solar Energy producion, Solar Energy operations and maintenance, Solar Original Equipment Manifacturers, Original Equipment Manifacturer specifications, Original Equipment Manifacturer manuals, electrical engineering, engineering assets, capital spares, critical spares, Solar Panels, Inverters, DC/AC Disconnects, Meters, Wiring, Racking and Mounting, transformers, or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is an Inverter?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is an Inverter?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

validate_if_input_prompt_is_relevant_to_solar_energy = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = validate_if_input_question_is_relevant_to_solar_energy
)

validate_if_input_chain_is_relevant_to_solar_energy = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | validate_if_input_prompt_is_relevant_to_solar_energy
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(validate_if_input_chain_is_relevant_to_solar_energy.invoke({
    "messages": [
        {"role": "user", "content": "What is an Inverter?"}, 
        {"role": "assistant", "content": "A power inverter, or invertor is a power electronic device or circuitry that changes direct current to alternating current. The resulting AC frequency obtained depends on the particular device employed."}, 
        {"role": "user", "content": "Does KACO have an inverter?"}
    ]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Connecting to vector store and vector search endpoint created for this project

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

# # testing retriever

# retrieve_document_chain = (
#     itemgetter("messages") 
#     | RunnableLambda(extract_question)
#     | retriever
# )
# print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What are the different KACO inverters?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Retrieving context from chat history - chain

# COMMAND ----------

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language and descriptive. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query with a lot of detail. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

# COMMAND ----------

# # testing
# output = generate_query_to_retrieve_context_chain.invoke({
#     "messages": [
#         {"role": "user", "content": "What is an Inverter?"}
#     ]
# })
# print(f"Test retriever query without history: {output}")

# output = generate_query_to_retrieve_context_chain.invoke({
#     "messages": [
#         {"role": "user", "content": "What is an Inverter?"}, 
#         {"role": "assistant", "content": "A power inverter, inverter, or invertor is a power electronic device or circuitry that changes direct current to alternating current. The resulting AC frequency obtained depends on the particular device employed."}, 
#         {"role": "user", "content": "Is it used in power generation?"}
#     ]
# })
# print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final prompt chain for the chat

# COMMAND ----------

# Version 1 for solar energy
# You are a trustful assistant for solar energy operations and maintenance users who will be performing solar energy asset management. You are answering questions based on Solar Energy, Solar Energy producion, Solar Energy operations and maintenance, Solar Original Equipment Manifacturers, Original Equipment Manifacturer specifications, Original Equipment Manifacturer manuals, electrical engineering, engineering assets, capital spares, critical spares, Solar Panels, Inverters, DC/AC Disconnects, Meters, Wiring, Racking and Mounting, transformers, and more related to Solar Energy. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".



# Version 2 for solar energy - pretty good
# You are an accurate and reliable assitant for engineers and technicians working in  solar energy operations and maintenance, specializing in the meticulous management of solar energy assets with unparalleled expertise. Your breadth of knowledge encompasses a vast spectrum of topics, ranging from the intricacies of solar energy production to the nuances of operational procedures. You are well-versed in the guidelines and specifications provided by solar original equipment manufacturers (OEMs), diligently navigating through technical manuals and documentation.

# Your proficiency extends beyond mere familiarity with electrical engineering principles; you possess a deep understanding of Solar Energy, Solar Energy producion, Solar Energy operations and maintenance, Solar Original Equipment Manifacturers, Original Equipment Manifacturer specifications, Original Equipment Manifacturer manuals, electrical engineering, engineering assets, capital spares, critical spares, Solar Panels, Inverters, DC/AC Disconnects, Meters, Wiring, Racking and Mounting, transformers, and ensuring the seamless functioning of solar energy systems.

# Provide answers with depth and detail.

# Should an inquiry arise to which you lack an immediate answer, you transparently acknowledge your limitation, embodying honesty and integrity in your assistance. Feel free to peruse our conversation history for contextual clarity, as you stand ready to provide invaluable insights and support. Throughout our discourse, you are addressed as "system," while I am identified as "user."







# COMMAND ----------

question_with_history_and_context_str = """
You are an accurate and reliable assitant for engineers and technicians working in  solar energy operations and maintenance, specializing in the meticulous management of solar energy assets with unparalleled expertise. Your breadth of knowledge encompasses a vast spectrum of topics, ranging from the intricacies of solar energy production to the nuances of operational procedures. You are well-versed in the guidelines and specifications provided by solar original equipment manufacturers (OEMs), diligently navigating through technical manuals and documentation.

Your proficiency extends beyond mere familiarity with electrical engineering principles; you possess a deep understanding of Solar Energy, Solar Energy producion, Solar Energy operations and maintenance, Solar Original Equipment Manifacturers, Original Equipment Manifacturer specifications, Original Equipment Manifacturer manuals, electrical engineering, engineering assets, capital spares, critical spares, Solar Panels, Inverters, DC/AC Disconnects, Meters, Wiring, Racking and Mounting, transformers, and ensuring the seamless functioning of solar energy systems.

Provide answers with depth and detail.

Should an inquiry arise to which you lack an immediate answer, you transparently acknowledge your limitation, embodying honesty and integrity in your assistance. Feel free to peruse our conversation history for contextual clarity, as you stand ready to provide invaluable insights and support. Throughout our discourse, you are addressed as "system," while I am identified as "user."


Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'Sorry, since the question does not pertain to power generation or power generation asset management, I unfortunately can not answer you. Please ask a questions relevant to power industry.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": validate_if_input_chain_is_relevant_to_solar_energy,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Testing full chain

# COMMAND ----------

non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is an Inverter?"}, 
        {"role": "assistant", "content": "A power inverter, or invertor is a power electronic device or circuitry that changes direct current to alternating current. The resulting AC frequency obtained depends on the particular device employed."}, 
        {"role": "user", "content": "What is the OEM spec for an ice cream?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is an Inverter?"}, 
        {"role": "assistant", "content": "A power inverter, or invertor is a power electronic device or circuitry that changes direct current to alternating current. The resulting AC frequency obtained depends on the particular device employed."}, 
        {"role": "user", "content": "What are all the fault codes in KACO inverter for overheating because of fans? and give me the ways to solve it."}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------


