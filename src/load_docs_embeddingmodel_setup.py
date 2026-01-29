from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# load docs
# create chunks
# create vector embeddings
# store embedding in chroma/fiass

####


load_dotenv()
os.chdir("../")
data_path = "data"


# load docs
def load_docs(data_path):
    loader = DirectoryLoader(
        data_path, glob="*.pdf", loader_cls=PyPDFLoader  # type: ignore
    )
    documents = loader.load()
    return documents


documents = load_docs(data_path)
# print(documents[:2])
# print(len(documents))


def filter_extracted_docs_content(docs):
    return [
        Document(
            page_content=doc.page_content,
            metadata={
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            },
        )
        for doc in docs
    ]


filtered_document = filter_extracted_docs_content(documents)


# create chunks


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


chunked_documents = create_chunks(filtered_document)


