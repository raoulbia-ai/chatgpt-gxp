"""
https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a.html#multi-document-queries
"""
import streamlit as st
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

import asyncio
import nest_asyncio
nest_asyncio.apply()


# from langchain import OpenAI # deprecated
from langchain_openai import OpenAI

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import set_global_service_context
from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine, RouterQueryEngine

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")

import httpx
with httpx.Client(verify=False) as client:
    response = client.get('http://localhost:8501/')


# text-davinci=003 is deprecated - see https://stackoverflow.com/questions/77789886/openai-api-error-the-model-text-davinci-003-has-been-deprecated
# llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct", max_tokens=-1)

service_context = ServiceContext.from_defaults(
    # llm=llm
)
set_global_service_context(service_context=service_context)



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
def main():

    state = get_session_state()

    # Check if there's an ongoing conversation
    if not hasattr(state, 'conversation'):
        state.conversation = []

    # Check if there's an ongoing conversation
    if hasattr(state, 'conversation'):
        conversation = state.conversation
    else:
        conversation = []

    # Get the number of questions asked
    num_questions = len([speaker for speaker, _ in conversation if speaker == 'You'])

    # Create an input box for the next question
    user_input = st.text_input(f'Enter your question {num_questions+1}:', key=f'question{num_questions+1}')

    # Create a button to confirm the input
    if st.button('Ask Me!', key=f'button{num_questions+1}'):
        # Add the question to the conversation
        conversation.append(('You', user_input))

        # Update the conversation in the session state
        set_session_state(conversation=conversation)

        prompt = f"""
        You have been trained on the following documents:

        1. FDA_Title21_CFR_Part_11_computer_systems.pdf
        2. FDA_GAMP5.pdf
        3. EU_annex11_computerised_systems.pdf
        4. EMA_cloud_strategy.pdf
        5. EMA_guideline_q9_quality_risk_management.pdf

        Make sure to use all of them when answering the question below.
        You may use external knowledge to reason about the question you are asked to answer but DO NOT use external knowledge to answer the question.

        QUESTION:
        {user_input}
        """

        response = query_engine.query(prompt)

        # Extract the answer without metadata
        answer = get_response_without_metadata(response)

        # Add the answer to the conversation
        conversation.append(('AI', answer))

        # Update the conversation in the session state
        set_session_state(conversation=conversation)

    # Display the conversation
    for speaker, text in conversation:
        st.write(f'{speaker}: {text}')


if __name__ == "__main__":
    main()