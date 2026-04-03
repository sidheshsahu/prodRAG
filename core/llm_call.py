from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def llm_1():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
    return llm


def llm_2():
    llm = OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        generation_kwargs={"max_tokens": 512},
    )
    return llm


def llm_3():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    return llm
