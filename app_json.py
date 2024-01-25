"""
https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a.html#multi-document-queries

EMA_cloud_strategy
EMA_guideline_q9_quality_risk_management
EU_annex11_computerised_systems
FDA_GAMP5
FDA_Title21_CFR_Part_11_computer_systems
OSS_in_Regulated_Industries_based_on_GAMP

What are the key principles of GAMP 5 for ensuring data integrity in GxP systems?
How does GAMP 5 guide address risk management in pharmaceutical manufacturing?
Explain the role of quality systems standards, like ISO 9000, in GAMP 5 compliant systems.
What are the best practices for implementing a life cycle approach to GxP computerized systems according to GAMP 5?
Describe the GAMP 5 recommendations for leveraging supplier involvement in GxP system development.
"""
import streamlit as st
# from streamlit.report_thread import get_report_ctx
# from streamlit.server.server import Server

# import asyncio
import nest_asyncio
nest_asyncio.apply()


# from langchain import OpenAI # deprecated
# from langchain_openai import OpenAI

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import set_global_service_context
# from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine, RouterQueryEngine

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")
username = os.environ.get('BASIC_AUTH_USERNAME')
password = os.environ.get('BASIC_AUTH_PASSWORD')

from pathlib import Path
from llama_index import download_loader

JSONReader = download_loader("JSONReader")
loader = JSONReader()

service_context = ServiceContext.from_defaults(
    # llm=llm
)
set_global_service_context(service_context=service_context)


try:
    fda_comp_systems_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    fda_comp_systems_index = load_index_from_storage(fda_comp_systems_storage_context)

    fda_gxp_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    fda_gxp_index = load_index_from_storage(fda_gxp_storage_context)

    eu_comp_systems_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    eu_comp_systems_index = load_index_from_storage(eu_comp_systems_storage_context)

    ema_cloud_strategy_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    ema_cloud_strategy_index = load_index_from_storage(ema_cloud_strategy_storage_context)

    ema_risk_management_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    ema_risk_management_index = load_index_from_storage(ema_risk_management_storage_context)

    oss_storage_context = StorageContext.from_defaults(persist_dir='storageDefaultLlmAllJSON')
    oss_index = load_index_from_storage(oss_storage_context)

    print('loading from disk')
except:
    fda_comp_systems = loader.load_data(Path("./data/json/FDA_Title21_CFR_Part_11_computer_systems.json"))
    fda_gxp = loader.load_data(Path("./data/json/FDA_GAMP5.json"))
    eu_comp_systems = loader.load_data(Path("./data/json/EU_annex11_computerised_systems.json"))
    ema_cloud_strategy = loader.load_data(Path("./data/json/EMA_cloud_strategy.json"))
    ema_risk_management = loader.load_data(Path("./data/json/EMA_guideline_q9_quality_risk_management.json"))
    oss = loader.load_data(Path("./data/json/OSS_in_Regulated_Industries_based_on_GAMP.json"))

    print(f'Loaded  fda_comp_systems with {len(fda_comp_systems)} pages')
    print(f'Loaded  fda_gxp with {len(fda_gxp)} pages')
    print(f'Loaded  eu_comp_systems with {len(eu_comp_systems)} pages')
    print(f'Loaded  ema_cloud_strategy with {len(ema_cloud_strategy)} pages')
    print(f'Loaded  ema_risk_management with {len(ema_risk_management)} pages')
    print(f'Loaded  ema_risk_management with {len(oss)} pages')

    fda_comp_systems_index = VectorStoreIndex.from_documents(fda_comp_systems, show_progress=True)
    fda_gxp_index = VectorStoreIndex.from_documents(fda_gxp, show_progress=True)
    eu_comp_systems_index = VectorStoreIndex.from_documents(eu_comp_systems, show_progress=True)
    ema_cloud_strategy_index = VectorStoreIndex.from_documents(ema_cloud_strategy, show_progress=True)
    ema_risk_management_index = VectorStoreIndex.from_documents(ema_risk_management, show_progress=True)
    oss_index = VectorStoreIndex.from_documents(oss, show_progress=True)

    fda_comp_systems_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')
    fda_gxp_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')
    ema_cloud_strategy_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')
    eu_comp_systems_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')
    ema_risk_management_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')
    oss_index.storage_context.persist(persist_dir='storageDefaultLlmAllJSON')

fda_comp_systems_engine = fda_comp_systems_index.as_query_engine(similarity_top_k=3)
fda_gxp_engine = fda_gxp_index.as_query_engine(similarity_top_k=3)
eu_comp_systems_engine = eu_comp_systems_index.as_query_engine(similarity_top_k=3)
ema_cloud_strategy_engine = ema_cloud_strategy_index.as_query_engine(similarity_top_k=3)
ema_risk_management_engine = ema_risk_management_index.as_query_engine(similarity_top_k=3)
oss_engine = oss_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=fda_comp_systems_engine,
        description="""This document outlines the requirements and regulations for electronic
        records and electronic signatures in the context of the Federal Food, Drug,and Cosmetic Act."""
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
        management principles and tools for making effective and consistent risk-based decisions in the pharmaceutical 
        industry and regulatory environment."""
    ),
    QueryEngineTool.from_defaults(
        query_engine=oss_engine,
        description="""The document is a guide on using Open Source Software (OSS) in regulated industries, particularly 
        focusing on compliance with Good Automated Manufacturing Practice (GAMP). It discusses the benefits, challenges, 
        and strategies for effectively integrating OSS in these industries, highlighting the importance of a risk-based 
        approach to validation and adherence to regulatory requirements. The guide also explores the similarities and 
        differences between OSS and commercial software, emphasizing the need for proper management and validation 
        processes to ensure OSS meets industry standards."""
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
            6. Guide for using Open Source Software (OSS) in Regulated Industries based on GAMP

            When responding to questions, provide a succinct summary, followed by a detailed analysis grounded in these 
            documents, and conclude with practical implications. Your explanations should be technically comprehensive, 
            tailored for an audience highly knowledgeable in the pharmaceutical field.  

            Maintain strict adherence to the content within these documents. In the case of ambiguities in user input, 
            seek clarification to ensure precise responses. While prioritizing direct answers, also proactively suggest 
            related topics or questions for deeper exploration when relevant.
            
            End your response by adding a paragraph referencing the name or names of the document or documents you have 
            used to answer the question.
            
            DO NOT use any external knowledge when answering the question!

            QUESTION:
            {user_input}
        """

        response = query_engine.query(prompt)
        print(response)
        answer = get_response_without_metadata(response)

        # Add answer to conversation
        conversation.append(("AI", answer))
        # Clear input box
        st.session_state.question_input = ""


if __name__ == "__main__":
    main()
