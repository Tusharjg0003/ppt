from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os

# Initialize FastAPI
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your front-end URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client
client = chromadb.Client()

# Load and split PDF document
loader = PyPDFLoader("Policies.pdf")
pdfData = loader.load()

text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=100)
splitData = text_splitter.split_documents(pdfData)

# Define storage for Chroma database
collection_name = "Ubi"
local_directory = "hammadrizwan"
persist_directory = os.path.join(os.getcwd(), local_directory)

# Set up OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM and embeddings
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4",
    temperature=0.7
)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Create Chroma vector database
vectDB = Chroma.from_documents(
    splitData,
    embeddings,
    collection_name=collection_name,
    persist_directory=persist_directory
)

# Deduplicate results
def deduplicate_results(results):
    unique_results = []
    seen_content = set()
    for doc in results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    return unique_results

# Define prompt template
template = """  
You are a HR Assistant, a knowledgeable and professional HR Assistant chatbot for Production Precision Trading, dedicated to providing exceptional support to employees. ...

Context:
{context}

Question:
{question}

Your Response:
"""  

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[str]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vectDB.similarity_search(state["question"], k=7)
    filtered_docs = deduplicate_results(retrieved_docs)
    return {"context": filtered_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Request and Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    state = {"question": request.question, "context": [], "answer": ""}
    try:
        result = graph.invoke(state)
        return QueryResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # Remove Streamlit testing
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
