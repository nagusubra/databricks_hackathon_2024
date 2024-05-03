# Databricks notebook source
# MAGIC %pip install -q mlflow[databricks]==2.10.1 tqdm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Import libraries and modules

# COMMAND ----------

# MAGIC %run /Workspace/Repos/subramanian.narayana.ucalgary@gmail.com/databricks_hackathon_2024/_resources/00-init-advanced $reset_all_data=false
# MAGIC

# COMMAND ----------

from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import mlflow
import os

# COMMAND ----------

# MAGIC %md
# MAGIC #Import asset_nav model

# COMMAND ----------

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
model_name = f"{catalog}.{db}.asset_nav_chatbot_model_version_1"
model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Chat Interface Functions

# COMMAND ----------

document_drive_links={
                        "BP150_Manual_Kaco blueplanet_ Event codes - Vendor status codes - good.pdf": "https://drive.google.com/file/d/1WoNhXll_AKWL3ZFoEcNk0uoVBEcsdVVr/view?usp=sharing",
                        "BP150_Manual_Kaco blueplanet_1-9-2023 DCB - Manual extended - medium - 2.pdf": "https://drive.google.com/file/d/1CLxl3ueXyDxm60KLjA0Ak7V7p_YL0EkK/view?usp=sharing",
                        "BP150_Manual_Kaco blueplanet_1-9-2023 DCB - Quick guide - good.pdf": "https://drive.google.com/file/d/1IQa-_d6B8dVAd8Zad1LBqipeYQVD2iEI/view?usp=sharing",
                        "Data sheet Kaco blue panet 150 TL3 - medium - 2.pdf": "https://drive.google.com/file/d/1p803z6u4H9YOKbz74J5ZdkB-ISxuhhXb/view?usp=sharing",
                        "Frequently Asked Questions About Maintenance of Solar Sites - good.pdf": "https://drive.google.com/file/d/1vj1C8i1rkNRps7HjQwM5eXXJhem_DzB4/view?usp=sharing",
                        "MNL_BP_87.0-165TL3_DSCK-InstallManuel_210504_en - good - 2.pdf": "https://drive.google.com/file/d/1ON9H6suHiDdlnx6imITf9HDv0olqXgGe/view?usp=sharing",
                        "Solar Asset Management - good.pdf": "https://drive.google.com/file/d/1itXqAVm-H3HaG2wFk2CRHwmUZMhcVWsD/view?usp=sharing",
                        "Solar-Basics - good.pdf": "https://drive.google.com/file/d/1z9uIzO-BNfERmmlXBkbx20sqacALyDyN/view?usp=sharing"
}

# COMMAND ----------

box_shadow = "10px 10px 10px #0071c9"
user_background_color = "#b0dcff"
rag_background_color = "#43a8f7"

def user_message_html(message):
    return f"""
                <head>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

                    /* Custom CSS for chat bubbles */
                    .chat-container-user {{
                        width: 75%;
                        border-radius: 20px;
                        background: linear-gradient(to left, {user_background_color}, {rag_background_color});
                        padding: 20px;
                        box-shadow: {box_shadow};
                        margin-bottom: 20px;
                        margin-left: 20%;
                        margin-right: auto;
                        font-size: 16px;
                        color: white;
                        font-weight: bold;
                        font-family: 'Poppins', sans-serif;
                        position: relative;
                        overflow: hidden;
                    }}

                    .chat-bubble-user {{
                        content: '';
                        position: absolute;
                        top: 0;
                        right: 0;
                        width: 0;
                        height: 0;
                        border-top: 20px solid transparent;
                        border-left: 20px solid {user_background_color};
                        border-right: 20px solid transparent;
                        border-bottom: 20px solid transparent;
                    }}

                    .avatar-user {{
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        margin-left: 10px;
                    }}

                    .message-text-user {{
                        flex: 1;
                    }}
                </style>
                </head>

                <div class="chat-container-user">
                    <div style="display: flex; align-items: center;">
                        <div class="message-text-user">
                            { f"<h1><strong>You</strong> <br></h1>" +  message}
                        </div>
                        <img class="avatar-user" src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png?raw=true"/>
                    </div>
                </div>
            """

def assistant_message_html(message):
    return f"""
                <head>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

                    /* Custom CSS for chat bubbles */
                    .chat-container-assistant {{
                        width: 75%;
                        border-radius: 20px;
                        background: linear-gradient(to left, {rag_background_color}, {user_background_color});
                        padding: 20px;
                        box-shadow: {box_shadow};
                        margin-bottom: 20px;
                        margin-left: auto;
                        margin-right: 20%;
                        font-size: 16px;
                        color: white;
                        font-weight: bold;
                        font-family: 'Poppins', sans-serif;
                        position: relative;
                        overflow: hidden;
                    }}

                    .chat-bubble-assistant {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 0;
                        height: 0;
                        border-top: 20px solid transparent;
                        border-right: 20px solid {rag_background_color};
                        border-left: 20px solid transparent;
                        border-bottom: 20px solid transparent;
                    }}

                    .avatar-assistant {{
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        margin-right: 10px;
                    }}

                    .message-text-assistant {{
                        flex: 1;
                    }}
                </style>
                </head>

                <div class="chat-container-assistant">
                    <div style="display: flex; align-items: center;">
                        <img class="avatar-assistant" src="https://e7.pngegg.com/pngimages/498/917/png-clipart-computer-icons-desktop-chatbot-icon-blue-angle-thumbnail.png?raw=true"/>
                        <div class="message-text-assistant">
                            {"<h1><strong>Asset Nav Assistant</strong> <br></h1>" + message}
                        </div>
                    </div>
                </div>
            """

# COMMAND ----------

def display_chat(chat_history):
    chat_history_html = "".join(
        [
            user_message_html(m["content"])
            if m["role"] == "user"
            else assistant_message_html(m["content"])
            for m in chat_history
        ]
    )
    displayHTML(chat_history_html)

def get_response_in_html_format(response):
    sources_set = set(response["sources"])
    sources_html = (
        "<br/><br/><strong>Sources:</strong><br/> <ul>"
        + "\n".join(
            [f"""<li><a href="{document_drive_links[s[65:]].replace(" ","%20")}">{s[65:]}</a></li>""" for s in sources_set]
        )
        + "</ul>"
    ) if response["sources"] else ""
    answer = response["result"].replace("\n", "<br/>")
    response_html = f"""{answer}{sources_html}"""
    return response_html

def chatbot_interface(user_input, dialog_history_for_rag, dialog_history_for_user):

    dialog_history_for_rag["messages"].append({"role": "user", "content": user_input})
    dialog_history_for_user["messages"].append({"role": "user", "content": user_input})
    
    # print(f'Testing with relevant history and question...')

#     response = {'result': "172 Internal fan error: This fault code is displayed when there is a failure of an internal fan or the corresponding tacho signal. The power is reduced to 50% Pnom and all 3 LEDs light up on the device.\n\nCauses and solutions for this fault code:\n\n1. Fan blocked: Check if the fan is blocked by debris or foreign particles. If yes, clean it carefully.\n2. Plugs not correctly plugged in: Ensure that the plugs are correctly plugged in.\n\nAdditionally, to prevent overheating due to fans, follow these guidelines:\n\n1. Regularly clean the inverter and its surrounding area to avoid dust accumulation.\n2. Ensure proper ventilation around the inverter to allow heat dissipation.\n3. Regularly inspect the fans for wear and tear and replace them if necessary.\n4. Periodically check the electrical connections and wiring for loose connections or damage.\n5. Follow the manufacturer's recommended maintenance schedule for the inverter.",
#  'sources': ['dbfs:/Volumes/main/asset_nav/volume_oem_documentation/input_data/BP150_Manual_Kaco blueplanet_ Event codes - Vendor status codes - good.pdf',
#   'dbfs:/Volumes/main/asset_nav/volume_oem_documentation/input_data/BP150_Manual_Kaco blueplanet_1-9-2023 DCB - Manual extended - medium - 2.pdf',
#   'dbfs:/Volumes/main/asset_nav/volume_oem_documentation/input_data/BP150_Manual_Kaco blueplanet_ Event codes - Vendor status codes - good.pdf',
#   'dbfs:/Volumes/main/asset_nav/volume_oem_documentation/input_data/BP150_Manual_Kaco blueplanet_1-9-2023 DCB - Manual extended - medium - 2.pdf']}
    
    response = rag_model.invoke(dialog_history_for_rag)

    dialog_history_for_rag["messages"].append({"role": "assistant", "content": response["result"]})

    response_html = get_response_in_html_format(response)
    dialog_history_for_user["messages"].append({"role": "assistant", "content": response_html})

    # display_chat(dialog_history_for_user["messages"])
    return dialog_history_for_user["messages"]

# COMMAND ----------

# MAGIC %md
# MAGIC #Chat Interface

# COMMAND ----------

dialog_history_for_rag  = {"messages": [{"role": "assistant", "content": "How can I help you?"}]}
dialog_history_for_user = {"messages": [{"role": "assistant", "content": "How can I help you?"}]}

question = "What is your purpose?"

dialog_history_for_user_messages = chatbot_interface(question, dialog_history_for_rag, dialog_history_for_user)
display_chat(dialog_history_for_user_messages)

# COMMAND ----------

# MAGIC %md
# MAGIC #Chat Interface - continous

# COMMAND ----------

dialog_history_for_rag  = {"messages": [{"role": "assistant", "content": "How can I help you?"}]}
dialog_history_for_user = {"messages": [{"role": "assistant", "content": "How can I help you?"}]}

while True:
    user_input = input("Enter your message (or 'end' to exit): ")
    if user_input.lower() == "end":
        break

    loading_widget = widgets.IntProgress(value=0, min=0, max=1, description='Loading:', bar_style='info', style={'bar_color': '#43a8f7'})
    display(loading_widget)

    dialog_history_for_user_messages = chatbot_interface(user_input, dialog_history_for_rag, dialog_history_for_user)

    clear_output()
    loading_widget = widgets.IntProgress(value=1, min=0, max=1, description='Completed:', bar_style='info', style={'bar_color': '#43a8f7'})
    display(loading_widget)

    display_chat(dialog_history_for_user_messages)

# COMMAND ----------

# MAGIC %md
# MAGIC #Dialog history sample

# COMMAND ----------

# dialog_history_for_rag

# COMMAND ----------

# dialog_history_for_user
