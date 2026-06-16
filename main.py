import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2")
BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
llm = Ollama(model=MODEL_NAME, base_url=BASE_URL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
class ChatRequest(BaseModel):
    message: str
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingests text content or email files directly into your growing custom brain."""
    try:
        contents = await file.read()
        text_content = contents.decode("utf-8")
        if not text_content.strip():
            return {"status": "error", "message": "The uploaded data content file is completely empty."}
        chunks = text_splitter.split_text(text_content)
        vector_db.add_texts(texts=chunks)        
        return {"status": "success", "message": f"Successfully loaded {len(chunks)} contextual items into memory!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Retrieves context pieces and crafts responses based strictly on your fed data."""
    user_query = request.message
    docs = vector_db.similarity_search(user_query, k=4)
    context = "\n---\n".join([doc.page_content for doc in docs])    
    rag_prompt = f"""You are a specialized AI knowledge system trained specifically on custom mail logs and user context records. 
Analyze the user query based strictly on the provided custom facts context memory. 
If the context does not contain relevant insights, declare clearly that the information is missing.
Context Memory:
{context}
Query: {user_query}
Answer:"""

    response = llm.invoke(rag_prompt)
    return {"response": response}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)