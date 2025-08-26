import json
import re
import argparse
from typing import List, Dict, Optional

# LangChain core + community
from langchain.agents import initialize_agent, Tool
from langchain_core.tools import retriever
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# Pydantic for schema validation
from numpy._core.numeric import result_type
from pydantic import BaseModel, Field, ValidationError


# --------------------------- JSON Schema ---------------------------
class Standard(BaseModel):
    id: str = Field(
        ...,
        description="Standard identifier (e.g., 'NH.SCI.5.LS2.1')",
        min_length=1,
        pattern=r"^[A-Za-z0-9._:\-]+$",
    )
    description: str = Field(..., min_length=5, description="Plain-language description of the standard")
    grade_band: Optional[str] = Field(
        None,
        description="Optional grade band label (e.g., '3-5', '6-8')"
    )

class LessonPlan(BaseModel):
    objective: str = Field(..., min_length=10, description="A clear learning objective for the lesson")
    activities: List[str] = Field(..., min_items=1, description="Step-by-step student-facing activities")
    assessment: str = Field(..., min_length=5, description="How understanding will be checked")
    standards: List[Standard] = Field(..., min_items=1, description="Aligned standards with IDs and descriptions")
    supplemental_resources: List[Dict[str, str]] = Field(
        default_factory=list, description="List of {'title','url'}"
    )


try:
    JSON_SCHEMA = LessonPlan.model_json_schema()  # Pydantic v2
except AttributeError:
    JSON_SCHEMA = LessonPlan.schema()  # Pydantic v1

json_parser = JsonOutputParser(pydantic_object=LessonPlan)
format_instructions = json_parser.get_format_instructions()

# --------------------------- Tools ---------------------------

@tool
def lesson_planner_tool(topic: str) -> str:
    """
    Generate a lesson plan (json) given instructions and any retrieved standards context included in the prompt.
    """
    llm = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a curriculum designer. Use the provided context to produce a lesson plan strictly "
                "following the requested JSON schema. Do not include any supplemental resources."
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Topic: {topic}\n\n"
                "Follow these format rules:\n{format_instructions}"
            ),
        ]
    )

    loader = TextLoader('data/nh_standards.txt')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory='./nh_chroma')
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})


    chain = (
    {"context": retriever, "topic": RunnablePassthrough()}
    | prompt.partial(format_instructions=format_instructions)
    | llm
    | json_parser  # <- parses into a validated LessonPlan Pydantic object
    )
    return chain.invoke(topic)  # returns str


@tool
def web_search_tool(query: str) -> str:
    """
    Returns a list of resources as a JSON string.
    """
    llm = OllamaLLM(model="llama3")
    prompt = PromptTemplate.from_template("""Extract and return the most important keyword from this query: {query} 
    Return only the keyword, no explanation or unnecessary words""")
    chain = prompt | llm
    keyword = chain.invoke(query)
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    results = search.run(keyword, results=2)
    print(type(results))
    print(results)
    return results


@tool
def json_formatter_tool(inputs: str) -> str:
    """
    Returns a JSON string matching the LessonPlan schema.
    """
    HUMAN = """Build the FINAL lesson plan JSON using the format rules and the provided content.

        Follow these format rules (do not echo anything else):
        {format_instructions}

        INPUT:
        {inputs}

        Return ONLY the JSON object.
    """
    llm = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous JSON composer that outputs ONLY valid JSON matching the schema."
            ),
            (
                "human",
                HUMAN
            ),
        ]
    )

    chain = prompt.partial(format_instructions=format_instructions) | llm | json_parser
    lesson_plan = chain.invoke({'inputs':inputs})

    return lesson_plan


# --------------------------- Agent Runner ---------------------------
def run_agent(topic: str, verbose: bool = True) -> str:
    """
    Runs a ReAct-style agent that decides which tools to call and in what order.
    Returns a JSON string matching the LessonPlan schema.
    """

    llm = OllamaLLM(model="llama3")

    tools = [
        Tool.from_function(func=lesson_planner_tool,name='LessonPlanner',description='Generate a draft lesson plan (text) given instructions and any retrieved standards context included in the prompt.'),
        Tool.from_function(func=web_search_tool,name='WebSearch',description=''),
        Tool.from_function(func=json_formatter_tool,name='JSONFormatter',description='',return_direct=True),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=verbose,
        handle_parsing_errors=True,
    )

    # High-level instruction to encourage proper tool use and final JSON via JSONFormatter
    user_task = f"""
You are a curriculum planning agent. The user topic is: "{topic}".
Steps you should take (but you decide the order):
1) Draft a plan with LessonPlanner using the topic and any standards context you have.
2) If activities/resources are requested or beneficial, call WebSearch with the topic to find classroom resources (2–5).
3) Call JSONFormatter last with a JSON payload follow these format rules:\n{format_instructions} If WebSearch returns any resources, add them to the supplemental_resources list - use Page value as the title and append the Page value to 'http://wikipedia.org/wiki/' as the url.
   
Return ONLY the final JSON produced by JSONFormatter.
"""

    result = agent.invoke(user_task)
    return result


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Planner Agent (RAG + Web Search + JSON output)")
    parser.add_argument("--topic", default="5th grade lesson plan for ecosystems.", help='e.g., "Fractions for 4th grade"')
    args = parser.parse_args()

    try:
        json_str = run_agent(topic=args.topic, verbose=True)
        print(json_str['output'])
    except Exception as e:
        print(json.dumps({"error": "Unhandled", "details": str(e)}, indent=2))
