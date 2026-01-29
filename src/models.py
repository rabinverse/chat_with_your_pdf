from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
load_dotenv()
model_groq = ChatGroq(model="openai/gpt-oss-120b")


def load_hf_model(repo_id, temperature_value, HF_TOKEN):
    model = HuggingFaceEndpoint(
        model=repo_id,
        temperature=temperature_value,
        model_kwargs={"token": HF_TOKEN, "max_length": 512},
    )
    return model
