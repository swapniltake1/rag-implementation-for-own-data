import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="SQL - Agent", layout="centered")
st.title("SQL Technical Assistant")
st.markdown("Ask questions based on the uploaded SQL documentation.")

# 1. Setup Logic (Cached so it only runs once)
@st.cache_resource
def init_rag_chain():
    load_dotenv()
    
    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    
    # Load Vector Store (Adjust path if necessary)
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # Initialize Local Model (Docker)
    llm = ChatOpenAI(
        model="ai/qwen3:0.6B-F16",
        base_url="http://localhost:12434/engines/v1",
        api_key="not-needed",
        temperature=0
    )
    
    system_prompt = (
        "You are a SQL expert. Use the following context to answer the question. "
        "Include code examples. If the answer isn't in the context, say so."
        "\n\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

# Initialize the chain
rag_chain = init_rag_chain()

# 2. Web Interface logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt_input := st.chat_input("Type your SQL question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            response = rag_chain.invoke({"input": prompt_input})
            full_response = response["answer"]
            
            # Extract Sources
            pages = list(set([doc.metadata.get('page', 'N/A') for doc in response['context']]))
            source_text = f"\n\n*Sources: Page(s) {', '.join(map(str, pages))}*"
            
            final_output = full_response + source_text
            st.markdown(final_output)
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final_output})