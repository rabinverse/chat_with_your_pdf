from dotenv import load_dotenv
# from connect_database import faiss_local
from prompt_design import return_custom_prompt_template
from models import model_groq
from langchain_community.vectorstores import FAISS
from embedding_model import get_embedding_model
from langchain_classic.chains.retrieval_qa.base import RetrievalQA



load_dotenv()


# embedding_model=get_embedding_model()
embedding_model = get_embedding_model()
db_faiss_path = "vectorstore/db_faiss"
db = FAISS.load_local(
    db_faiss_path, embedding_model, allow_dangerous_deserialization=True
)

# First, create the retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chain = RetrievalQA.from_chain_type(
    llm=model_groq,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": return_custom_prompt_template()},
)


while True:
    print("*" * 10)
    user_input = (
        input(
            "ask any medical questions\n Enter exit to stop \n",
        )
        .strip()
        .lower()
    )
    if user_input == "exit":
        break
    result = chain.invoke({"query":user_input})
    print(result)
