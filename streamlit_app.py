import streamlit as st
import csv
from datetime import datetime
from llama_index_setup import rerank_results, criteria, query_engine

def get_session_state():
    return st.session_state

def set_session_state(**kwargs):
    for key, value in kwargs.items():
        st.session_state[key] = value

def get_response_without_metadata(response):
    return response

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
        initial_results = query_engine.query(user_input)
        # reranked_results = rerank_results(initial_results, criteria)  # TODO fix bug
        reranked_results = initial_results
        
        # Get the current date and time
        now = datetime.now()
        # Format it as a string in the "yyyymmdd" format
        timestamp = now.strftime("%Y%m%d")
        
        # Save the question, the top answer, and the timestamp to a CSV file
        with open('questions_answers.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Write the question, the top answer, and the timestamp to the CSV file
            # Assuming reranked_results[0] is the top answer
            # writer.writerow([user_input, reranked_results[0], timestamp])
            writer.writerow([user_input, reranked_results, timestamp])
        
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