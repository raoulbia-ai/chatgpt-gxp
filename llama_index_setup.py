"""
this script can be used standalone to set up the persisted vector index store
"""
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

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # read local .env file
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.environ['OPENAI_API_KEY']

# load_dotenv(find_dotenv(), override=True)
# username = os.environ.get('BASIC_AUTH_USERNAME')
# password = os.environ.get('BASIC_AUTH_PASSWORD')

service_context = ServiceContext.from_defaults()
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

def rerank_results(results, criteria):
    """
    Reranks the results from the OpenAI LLM based on a given criteria.

    :param results: A list of results from the OpenAI LLM.
    :param criteria: A function that takes a result and returns a numerical score for reranking.
    :return: A list of reranked results.
    """
    try:
        # Sort the results based on the score returned by the criteria function.
        # The highest scores come first.
        reranked_results = sorted(results, key=criteria, reverse=True)
        return reranked_results
    except Exception as e:
        # In case of an error, you might want to handle it or log it.
        # Placeholder for future error handling/logging.
        raise e


def criteria(result):
    # This is a placeholder function. You'll need to replace this with your actual criteria.
    # For this example, we're assuming that each result has a 'score' attribute that we can use for reranking.
    return result['score']