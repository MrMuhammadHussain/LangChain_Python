from dotenv import load_dotenv
import os , time

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain ,SimpleSequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory


load_dotenv()

# API_KEY  = os.getenv('OPENAI_API_KEY')

def Practis():
    # Basic
    llm = OpenAI(temperature=0.3)
    print(llm.predict("write a Code to Make Server in python"))

# tasks = ["oman", "Tajikistan","Afghanistan","India"]
# PromptTemplate
    llm = OpenAI(temperature=0.2)
    promt = PromptTemplate.from_template("tell me About {Place}")
    # print (promt.format(Place="Pakistan"))
    chain = LLMChain(llm=llm, prompt=promt)
    # for task in tasks:
    print(chain.run("Pakistan"))
# SimpleSequentialChain
    promt = PromptTemplate.from_template("what the Name of Shop in karachi,Pakistan for {prodect}?")
    llm = OpenAI(temperature=0.3)
    chain1 = LLMChain(llm=llm, prompt=promt)

    promt = PromptTemplate.from_template("what are the Names of Flevors at {shop}?")
    llm = OpenAI(temperature=0.3)
    chain2 = LLMChain(llm=llm, prompt=promt)

    Chain = SimpleSequentialChain(chains=[chain1,chain2] ,verbose=True)
    print(Chain.run("CO2 Soda"))

    # SequentialChain
    # This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
    llm = OpenAI(temperature=1)
    template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""
    prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

    # This is an LLMChain to write a review of a play given a synopsis.
    llm = OpenAI(temperature=1)
    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    chain = SequentialChain(chains=[synopsis_chain, review_chain], input_variables=["era","title"],output_variables=["synopsis","review"],verbose=True)
    print(chain({"title":"Tragedy at sunset on the beach", "era": "Victorian England"}))

    # Action Agent
    llm = OpenAI(temperature=0.6)
    tools = load_tools(["wikipedia","llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # print(agent.run("how ilyas Qadri old in 2029 the date of brith is 1945 "))


    # Memory
    llm = OpenAI(temperature=0.7)
    promt = PromptTemplate.from_template("Tell me About {Location}")
    chain = LLMChain(llm=llm , prompt=promt , memory=ConversationBufferMemory() , verbose=True)
    print(chain.run("Pakistan"))
    print(chain.run("Chaina"))
    print(chain.run("India"))
    print(chain.run("Turkey"))
    print(chain.run("Iran"))
    print(chain.run("Oman"))
    print(chain.memory.buffer)