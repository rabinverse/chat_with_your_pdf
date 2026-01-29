from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model_groq = ChatGroq(model="openai/gpt-oss-120b")