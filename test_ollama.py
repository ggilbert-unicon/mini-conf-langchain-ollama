from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")
print(llm.invoke("What is LangChain?"))
