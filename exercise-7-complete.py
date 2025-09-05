import json
import random
from typing import List, Optional

# --- LangChain / Ollama / RAG imports ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Validation (Pydantic + JSON Schema) ---
from pydantic import BaseModel, Field, ValidationError

# --- RL --- 
from langchain_experimental.rl_chain.pick_best_chain import PickBest, PickBestFeatureEmbedder

try:
    # Optional: robust JSON Schema validation (pip install jsonschema)
    from jsonschema import Draft202012Validator, ValidationError as JSONSchemaValidationError
    HAVE_JSONSCHEMA = True
except Exception:
    HAVE_JSONSCHEMA = False


# ---------------------------
# 1) Define strict data model
# ---------------------------

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


# Export a JSON Schema for independent validation
try:
    # Pydantic v2
    JSON_SCHEMA = LessonPlan.model_json_schema()
except AttributeError:
    # Pydantic v1 fallback
    JSON_SCHEMA = LessonPlan.schema()


# ---------------------------
# 2) Build the RAG components
# ---------------------------

# Load your text corpus (update path as needed)
loader = TextLoader("data/nh_standards.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings + Vector store
embedding = OllamaEmbeddings(model="llama3")
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./nh_chroma")

# Retriever + LLM
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

# ---------------------------
# 3) Prompt with format rules
# ---------------------------

parser = JsonOutputParser(pydantic_object=LessonPlan)

format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a curriculum designer. Use the provided context to produce a lesson plan strictly "
            "following the requested JSON schema."
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Topic: {topic}\n\n"
            "Follow these format rules:\n{format_instructions}"
        ),
    ]
)

# Chain: pass topic to both retriever ("context") and prompt ("topic"), parse as LessonPlan
chain = (
    {"context": retriever, "topic": RunnablePassthrough()}
    | prompt.partial(format_instructions=format_instructions)
    | llm
    | parser  # <- parses into a validated LessonPlan Pydantic object
)

# ---------------------------
# 4) Run + Validate
# ---------------------------

if __name__ == "__main__":
    topic = "5th grade lesson plan for ecosystems."

    try:
        feature_embedder = PickBestFeatureEmbedder(auto_embed=True)
        # Define a scorer (e.g., a mock function that returns a score)
        def simple_scorer():
            # In a real scenario, this would be an actual metric.
            # For this example, we'll return a placeholder score.
            return random.random()

        rl_chain = PickBest(
            llm_chain=chain,
            selection_scorer=simple_scorer,
            feature_embedder=feature_embedder
        )

        output = rl_chain.invoke(topic)
        print(output)

    except Exception as e:
        # Catch-all for LLM/RAG/runtime errors
        print(
            json.dumps(
                {"error": "Unhandled error", "details": str(e)},
                indent=2
            )
        )
