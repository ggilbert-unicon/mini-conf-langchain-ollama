from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


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
vectordb.persist()

# Uses LLaMA 3 locally via Ollama
llm = Ollama(model="llama3")

# Uses Chroma to return the top 3 relevant chunks
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

template = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are a curriculum designer. Create a lesson plan for: {topic}.
        Include:
        - Objective
        - Activities
        - Assessment
        - Standards
        Respond in plain text.
        Context: {context}
    """)

question_answer_chain = create_stuff_documents_chain(llm, template)
chain = create_retrieval_chain(retriever, question_answer_chain)



query = "Create a 5th grade lesson plan for ecosystems."
result = chain.invoke({"topic": query})
print(result)
