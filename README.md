![Asset Nav Assistant - future app mock up](https://github.com/nagusubra/databricks_hackathon_2024/assets/52630559/de49baaa-b123-4172-91a3-160586141e70)
Asset Nav Assistant app mock up

## Inspiration

In the Energy and Manufacturing industry, a major hurdle that technicians and engineers face is that they spend valuable time grappling with extensive documentation to diagnose site issues. This bottleneck not only delays crucial work but also drains productivity. 

Our goal is to empower professionals to understand complex documents and get fast and accurate answers. By doing so, people can allocate their time more effectively and nurture an efficient and effective workforce.


## What it does

Our chatbot, Asset Nav Assistant, functions as an accurate Asset information assistant. It efficiently navigates vast documents to swiftly deliver accurate answers for users, saving proffesionals' time and effort. By eliminating the need to wait for document access or manually search through extensive files, it facilitates the completion of vital tasks seamlessly.
 
Moreover, we aim to develop a guide for new employees, this guide will offer more pertinent and current information, ensuring that new employees are equipped with the knowledge they need to succeed in their tasks.


## How we built it

In crafting Asset Nav Assistant, we employed Databricks and AWS as our foundational technologies and underwent the following phases:
 
- Preparation Phase: Parse, chunk, and store task specific data in vector format in Unity Catalog.
- Retrieval Phase: Employ the bge-large-en model for retrieval tasks.
- Augmentation Phase: Add related prompt templates to prepare a more comprehensive response.
- Generation Phase: Utilize the Mixtral 7B model to generate a response based on the retrieved and augmented information.
- Evaluation Phase: Evaluate the generated response to ensure its quality, readability, relevance, professionalism, and faithfulness. This iterative process allows us to fine-tune our model to achieve optimal performance.
- Interface Phase: Design an interface to allow users to interact with the final product within Databricks. This interface facilitates efficient access to information and enhances the overall user experience.

We employed Databricks and AWS as our foundational technologies. Harnessing the power of industry-leading cloud providers such as AWS and Databricks, we constructed an end-to-end RAG pipeline. Our core Large Language Model for this pipeline is the Mixtral 7B model, complemented by the DBRX instruct model for evaluation purposes.


## Challenges we ran into

1. Evaluation Data for Model: Scaling to diverse energy types led to accumulating vast data volumes with limited datasets for evaluation. To tackle this, we're devising an explorative evaluation data generation pipeline employing Instruct LLMs. This pipeline will generate context-based questions and answers to enrich our evaluation process.
2. Model Evaluation Metrics: Choosing appropriate metrics poses a challenge due to the industry's highly specific data.
3. Cost and Compute management: Execution of text based data, running models, creating serving endpoints, etc are compute intensive and cost intensive. We plan to deploy performance improvements to increase efficency of the RAG pipeline to mitage the cost and compute risks.
4. Model deployement challenges: Due to limitations of workspace requirements, we faced deployment issue for the RAG pipeline. We plan to Gradio/Streamlit for the mobile/tablet application development but hindered due to deployment issues.
5. Databricks Trial Workspace Limitation: Constraints on workspace usage and advanced features limited us from exploring the next steps for our project, but we intend to resolve this issue to implement our future strategy.


## Accomplishments that we're proud of

1. Achieved solution development within a tight 2-week timeframe, employing rapid prototyping and iterative enhancements.
2. Successfully constructed an end-to-end RAG solution, primed for deployment via Databricks serving endpoints.
3. Established a proof of concept aimed at streamlining technicians' and engineers' workflows, significantly reducing document access time and consolidating information from diverse data sources.
4. We developed an end-to-end RAG solution tailored to our dataset, finely tuned based on evaluation metrics, and equipped with UI interface within Databricks now fully prepared for deployment and scaling to different energy types.


## What we learned

1. Enhancing the RAG pipeline through fine-tuning substantially enhances output accuracy and performance.
2. Leveraging AWS services within the integrated Databricks platform proved to be a powerful combination.
3. Formulating a strategic deployment plan for Gen-AI applications tailored to industry-specific use cases was a key learning point.



## What's next for Asset Nav Assistant

1. Our primary future goal for our Asset Nav Assistant is implementing an Annual Audit system to ensure model integrity. Subject Matter Experts from various sectors will review the RAG pipeline’s outputs during iterative improvements. The audit system aims to gather SME feedback and current data to enhance the robustness of our RAG pipeline.
2. Implementing mobile/tablet app
3. Improve re-ranking systems for retriever chain
4. Using LLama guard and similar models to increase security
5. Expanding to various energy types via code replication
6. Utilizing Disintegrated storage instead of DAS to improve efficiency of inference
7. Scaling with cloud services like Databricks serving endpoints, AWS Lex, Azure AI bot service, etc.
8. Finalizing automated RAG evaluation data generation by creating an evaluation data generation pipeline using Instruct LLMs to generate context-based questions and answers.
9. Integerate a CI/CD cycle for the RAG pipeline.


Project Presentation: https://www.youtube.com/watch?v=mV-gipsRPXE
