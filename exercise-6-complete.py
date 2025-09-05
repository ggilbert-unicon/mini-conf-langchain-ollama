from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Loads your plain text file into a LangChain Document object
loader = TextLoader("data/nh_standards.txt")
documents = loader.load()

# Breaks the text into overlapping chunks so the LLM can process them better.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
# This uses Ollama locally to generate embeddings from each chunk — using the same model we’ll query later.
embedding = OllamaEmbeddings(model="llama3")
# Now we store the embeddings in a Chroma vector store so we can search them later in the RAG pipeline.
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./nh_chroma")

# Uses LLaMA 3 locally via Ollama
llm = OllamaLLM(model="llama3")

# Uses Chroma to return the top 3 relevant chunks
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

template = ChatPromptTemplate(([
    ("system", """
        You are a curriculum designer. You create a lesson plan for the given topic that
        includes:
        - Objective
        - Activities
        - Assessment
        - Standards
        Respond in plain text.
    """),
    ("human", "Context: {context}\nTopic: {topic}"),
]))

rag_chain = (
    {"context": retriever, "topic": RunnablePassthrough()}
    | template
    | llm  # Assume 'llm' is a configured language model instance
    | StrOutputParser()
)

config = RailsConfig.from_path("./guardrails.yml")
# Wrap the LangChain component with the guardrails
guardrails = RunnableRails(config)
chain_with_guardrails = guardrails | rag_chain


# 3. Invoke the chain with a user's question
topic = "5th grade lesson plan for ecosystems."
response = chain_with_guardrails.invoke(topic)

print(response)
