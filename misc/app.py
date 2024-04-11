"""
https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a.html#multi-document-queries
"""
import csv
from datetime import datetime
import streamlit as st
# from streamlit.report_thread import get_report_ctx
# from streamlit.server.server import Server

# import asyncio
import nest_asyncio
nest_asyncio.apply()

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine, RouterQueryEngine
from llama_index.core import (SimpleDirectoryReader,
                              Document,
                              VectorStoreIndex,
                              SummaryIndex,
                              load_index_from_storage,
                              StorageContext,
                              ServiceContext
                              )

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")
username = os.environ.get('BASIC_AUTH_USERNAME')
password = os.environ.get('BASIC_AUTH_PASSWORD')


# from pathlib import Path
# from llama_index import download_loader

# JSONReader = download_loader("JSONReader")
# loader = JSONReader()

service_context = ServiceContext.from_defaults(
    # llm=llm
)
# set_global_service_context(service_context=service_context)



try:
    fda_comp_systems_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAll')
    fda_comp_systems_index = load_index_from_storage(fda_comp_systems_storage_context)

    fda_gxp_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAll')
    fda_gxp_index = load_index_from_storage(fda_gxp_storage_context)

    eu_comp_systems_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAll')
    eu_comp_systems_index = load_index_from_storage(eu_comp_systems_storage_context)

    ema_cloud_strategy_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAll')
    ema_cloud_strategy_index = load_index_from_storage(ema_cloud_strategy_storage_context)

    ema_risk_management_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAll')
    ema_risk_management_index = load_index_from_storage(ema_risk_management_storage_context)

    print('loading from disk')
except:
    fda_comp_systems = SimpleDirectoryReader(input_files=[r"./data/FDA_Title21_CFR_Part_11_computer_systems.pdf"]).load_data()
    fda_gxp = SimpleDirectoryReader(input_files=["./data/FDA_GAMP5.pdf"]).load_data()
    eu_comp_systems = SimpleDirectoryReader(input_files=["./data/EU_annex11_computerised_systems.pdf"]).load_data()
    ema_cloud_strategy = SimpleDirectoryReader(input_files=["./data/EMA_cloud_strategy.pdf"]).load_data()
    ema_risk_management = SimpleDirectoryReader(input_files=["./data/EMA_guideline_q9_quality_risk_management.pdf"]).load_data()

    # fda_comp_systems = loader.load_data(Path("./data/json/FDA_Title21_CFR_Part_11_computer_systems.json"))
    # fda_gxp = loader.load_data(Path("./data/json/FDA_GAMP5.json"))
    # eu_comp_systems = loader.load_data(Path("./data/json/EU_annex11_computerised_systems.json"))
    # ema_cloud_strategy = loader.load_data(Path("./data/json/EMA_cloud_strategy.json"))
    # ema_risk_management = loader.load_data(Path("./data/json/EMA_guideline_q9_quality_risk_management.json"))
    
    print(f'Loaded  fda_comp_systems with {len(fda_comp_systems)} pages')
    print(f'Loaded  fda_gxp with {len(fda_gxp)} pages')
    print(f'Loaded  eu_comp_systems with {len(eu_comp_systems)} pages')
    print(f'Loaded  ema_cloud_strategy with {len(ema_cloud_strategy)} pages')
    print(f'Loaded  ema_risk_management with {len(ema_risk_management)} pages')

    fda_comp_systems_index = VectorStoreIndex.from_documents(fda_comp_systems, show_progress=True)
    fda_gxp_index = VectorStoreIndex.from_documents(fda_gxp, show_progress=True)
    eu_comp_systems_index = VectorStoreIndex.from_documents(eu_comp_systems, show_progress=True)
    ema_cloud_strategy_index = VectorStoreIndex.from_documents(ema_cloud_strategy, show_progress=True)
    ema_risk_management_index = VectorStoreIndex.from_documents(ema_risk_management, show_progress=True)

    fda_comp_systems_index.storage_context.persist(persist_dir='storageDefaultLlmAll')
    fda_gxp_index.storage_context.persist(persist_dir='storageDefaultLlmAll')
    ema_cloud_strategy_index.storage_context.persist(persist_dir='storageDefaultLlmAll')
    eu_comp_systems_index.storage_context.persist(persist_dir='storageDefaultLlmAll')
    ema_risk_management_index.storage_context.persist(persist_dir='storageDefaultLlmAll')

fda_comp_systems_engine = fda_comp_systems_index.as_query_engine(similarity_top_k=3)
fda_gxp_engine = fda_gxp_index.as_query_engine(similarity_top_k=3)
eu_comp_systems_engine = eu_comp_systems_index.as_query_engine(similarity_top_k=3)
ema_cloud_strategy_engine = ema_cloud_strategy_index.as_query_engine(similarity_top_k=3)
ema_risk_management_engine = ema_risk_management_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=fda_comp_systems_engine,
        description="""This document outlines the requirements and regulations for electronic
                              records and electronic signatures in the context of the Federal Food, Drug,
                              and Cosmetic Act."""
    ),
    QueryEngineTool.from_defaults(
        query_engine=fda_gxp_engine,
        description="""The FDA GXP document provides guidelines and regulations for ensuring the
                              quality, safety, and efficacy of pharmaceutical products through good manufacturing
                              practices, good laboratory practices, and good clinical practices."""
    ),
    QueryEngineTool.from_defaults(
        query_engine=eu_comp_systems_engine,
        description="""The European Union Annex 11 document provides guidelines for the use of
                              computerized systems in the pharmaceutical industry, ensuring their integrity,
                              reliability, and compliance with regulatory requirements."""
    ),
    QueryEngineTool.from_defaults(
        query_engine=ema_cloud_strategy_engine,
        description="""The EMEA cloud strategy document outlines guidelines for adopting
                              cloud-based technologies, focusing on security, data protection, compliance, and
                              accelerating digitalization and innovation initiatives"""
    ),
    QueryEngineTool.from_defaults(
        query_engine=ema_risk_management_engine,
        description="""The EMA risk management document provides guidance on quality risk
                              management principles and tools for making effective and consistent risk-based decisions
                              in the pharmaceutical industry and regulatory environment."""
    )
]

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

# Function to get the session state
def get_session_state():
    return st.session_state

# Function to set the session state
def set_session_state(**kwargs):
    for key, value in kwargs.items():
        st.session_state[key] = value

# Function to get the response without metadata
def get_response_without_metadata(response):
    # print(type(response))
    # print(response)
    return response #response['choices'][0]['text']

# Main app
import streamlit as st


def main():
    st.title("ChatGPT - GxP")
    st.write("<style>div.block-container{align-items: center;}</style>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center'>Proof of Concept ChatGPT Application trained on GAMP 5 and other similar public documents.</div>",
        unsafe_allow_html=True)

    # Session state to store conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Input for questions
    user_input = st.text_input("Enter your question:", key='question_input', on_change=handle_input,
                               args=(st.session_state.conversation,))

    # Display conversation
    for speaker, text in st.session_state.conversation:
        st.write(f"{speaker}: {text}")


def handle_input(conversation):
    user_input = st.session_state.question_input
    if user_input:

        # Add question to conversation
        conversation.append(("You", user_input))
        # Process the question and generate answer (placeholder)
        prompt = f"""
            As an AI expert in GxP regulatory guidelines and pharmaceutical compliance, your knowledge is exclusively 
            based on these specific documents:

            1. FDA Title 21 CFR Part 11: Computer Systems Validation in GxP Environments
            2. FDA's GAMP 5 Guide: Standards for GxP Compliant Computerized Systems
            3. EU's Annex 11: Computerised Systems in GxP Contexts
            4. EMA's Cloud Strategy: Compliance in Digital Data Storage for GxP
            5. EMA's Guideline on Quality Risk Management (Q9)

            When responding to questions, provide a succinct summary, followed by a detailed analysis grounded in these 
            documents, and conclude with practical implications. Your explanations should be technically comprehensive, 
            tailored for an audience highly knowledgeable in the pharmaceutical field.

            Maintain strict adherence to the content within these documents. In the case of ambiguities in user input, 
            seek clarification to ensure precise responses. While prioritizing direct answers, also proactively suggest 
            related topics or questions for deeper exploration when relevant.
            
            If the user question is not specific, ask for clarification!
            
            QUESTION:
            {user_input}
        """

        response = query_engine.query(prompt)
        answer = get_response_without_metadata(response)

        # Get the current date and time
        now = datetime.now()
        # Format it as a string in the "yyyymmdd" format
        timestamp = now.strftime("%Y%m%d")
        
        # Save the question, the top answer, and the timestamp to a CSV file
        with open('questions_answers.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Write the question, the top answer, and the timestamp to the CSV file
            # Assuming reranked_results[0] is the top answer
            writer.writerow([user_input, answer, timestamp])
        

        # Add answer to conversation
        conversation.append(("AI", answer))
        # Clear input box
        st.session_state.question_input = ""


if __name__ == "__main__":
    main()
