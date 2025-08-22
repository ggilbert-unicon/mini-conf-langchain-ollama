import json
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

class ___________(BaseModel):
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
loader = ___________("data/nh_standards.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings + Vector store
embedding = ______________(model="llama3")
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./nh_chroma")

# Retriever + LLM
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="___________")

# ---------------------------
# 3) Prompt with format rules
# ---------------------------

parser = JsonOutputParser(pydantic_object=____________)

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
            "Follow these format rules:\n{_______________}"
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
        # Invoke returns a LessonPlan (already Pydantic-validated)
        lesson_plan_dict: LessonPlan = chain.invoke(topic)
        print(lesson_plan_dict)
        lesson_plan = LessonPlan.model_validate(lesson_plan_dict)
        # Optional: Independent JSON Schema validation (extra safety)
        if HAVE_JSONSCHEMA:
            Draft202012Validator(JSON_SCHEMA).validate(json.loads(lesson_plan.model_dump_json()))

        # Print pretty JSON
        # Pydantic v2 uses model_dump_json; v1 uses json()
        try:
            print(lesson_plan.model_dump_json(indent=2))
        except AttributeError:
            print(lesson_plan.json(indent=2, ensure_ascii=False))

    except ValidationError as ve:
        # Pydantic validation errors (structure/type/constraints)
        print(
            json.dumps(
                {"error": "Pydantic validation failed", "details": json.loads(ve.json())},
                indent=2
            )
        )
    except HAVE_JSONSCHEMA and JSONSchemaValidationError as je:  # type: ignore
        # JSON Schema validation errors (only if jsonschema installed)
        print(
            json.dumps(
                {"error": "JSON Schema validation failed", "details": str(je)},
                indent=2
            )
        )
    except Exception as e:
        # Catch-all for LLM/RAG/runtime errors
        print(
            json.dumps(
                {"error": "Unhandled error", "details": str(e)},
                indent=2
            )
        )
