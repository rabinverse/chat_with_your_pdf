from langchain_huggingface import HuggingFaceEmbeddings
import os


def setup_hf_cache():
    base_dir = os.getcwd()
    hf_home = os.path.join(base_dir, "embedding_model")
    os.makedirs(hf_home, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_home

    return hf_home


# create vector embeddings
##############
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


hf_cache_path = setup_hf_cache()
