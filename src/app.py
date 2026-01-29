from dotenv import load_dotenv

# from connect_database import faiss_local
from prompt_design import return_custom_prompt_template
from models import model_groq
from langchain_community.vectorstores import FAISS
from embedding_model import get_embedding_model
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import streamlit as st
from dotenv import load_dotenv


st.title("Medical Chatbot")

prompt = st.chat_input("Ask any medical questions")


load_dotenv()


@st.cache_resource
def get_vector_store():

    # embedding_model=get_embedding_model()
    embedding_model = get_embedding_model()
    db_faiss_path = "vectorstore/db_faiss"
    db = FAISS.load_local(
        db_faiss_path, embedding_model, allow_dangerous_deserialization=True
    )
    return db


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt:
    st.chat_message("human").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

###
try:
    vector_store = get_vector_store()
    if vector_store is None:
        st.error("Failed to load vector store")
    chain = RetrievalQA.from_chain_type(
        llm=model_groq,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": return_custom_prompt_template()},
    )
    response = chain.invoke({"query": prompt})
    result = response["result"]
    # result_to_show=result+str(response["source_documents"])

    pages = sorted(
        {
            doc.metadata.get("page")
            for doc in response["source_documents"]
            if "page" in doc.metadata
        }
    )

    result_to_show = f"{result}\n\nPages: {', '.join(map(str, pages))}"

    st.chat_message("ai").markdown(result_to_show)
    st.session_state.messages.append({"role": "ai", "content": result})

except Exception as e:
    st.error(f"Error : {str(e)}")


###
