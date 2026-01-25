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


# create vector embeddings


def setup_hf_cache():
    base_dir = os.getcwd()
    hf_home = os.path.join(base_dir, "embedding_model")
    os.makedirs(hf_home, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_home

    return hf_home


##############
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


hf_cache_path = setup_hf_cache()
embedding_model = get_embedding_model()


# store embedding in chroma/fiass

DB_FIASS_PATH="vectorstore/db_fiass"
db=FAISS.from_documents(chunked_documents,embedding_model)
db.save_local(DB_FIASS_PATH)
