from dotenv import load_dotenv

from langchain.llms import OpenAI
import os, openai

load_dotenv()

API_KEY  = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0.3)
print(llm.predict("HI"))