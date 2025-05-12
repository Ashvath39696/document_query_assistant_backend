import os
import json
import asyncio
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ─────────────────────── Setup FastAPI ────────────────────────── #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────── Load Users ───────────────────────────── #
with open("users.json") as f:
    users = json.load(f)

from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

# Maps UUID to ListMemory instance (reset when server restarts)
session_memory_store: Dict[str, ListMemory] = {}


ROLE_HIERARCHY = {
    "hr_junior": ["common", "hr_junior"],
    "hr_mid": ["common", "hr_junior", "hr_mid"],
    "hr_senior": ["common", "hr_junior", "hr_mid", "hr_senior"],
    "finance": ["common", "hr_junior", "hr_mid", "hr_senior"]
}

ROLE_SOURCES = {
    "common": ["common_docs/easy/febbank.pdf"],
    "hr_junior": ["common_docs/centralbank_leave_policy.pdf"],
    "hr_mid": ["common_docs/easy/Business Analyst Offer Letter.pdf"],
    "hr_senior": []
}

# ─────────────────────── Auth & Models ────────────────────────── #
def get_user_by_uuid(uuid: str):
    for user in users:
        if user["uuid"] == uuid:
            return user
    return None

def validate_uuid(x_uuid: str = Header(...)):
    user = get_user_by_uuid(x_uuid)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid UUID")
    return user

class LoginRequest(BaseModel):
    name: str
    password: str

class AskRequest(BaseModel):
    query: str

# ─────────────────────── LangChain Chunker ────────────────────── #
class RoleBasedPDFChunker:
    def __init__(self, role_sources, persist_directory="chroma_db", force_reload=False):
        self.role_sources = role_sources
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.collections = {}
        self.chroma_collections = {}
        self.force_reload = force_reload
        os.makedirs(self.persist_directory, exist_ok=True)
        self._load_existing_collections()

    def _load_existing_collections(self):
        for role in self.role_sources:
            role_dir = os.path.join(self.persist_directory, role)
            if os.path.exists(role_dir) and not self.force_reload:
                try:
                    self.chroma_collections[role] = Chroma(
                        persist_directory=role_dir,
                        embedding_function=self.embeddings,
                        collection_name=f"{role}_collection"
                    )
                except Exception as e:
                    print(f"Could not load collection for {role}: {e}")

    def load_and_chunk_pdfs(self):
        for role, paths in self.role_sources.items():
            if role in self.chroma_collections and not self.force_reload:
                print(f"✅ Skipping PDF load for role '{role}' — collection exists.")
                continue
            chunks = []
            for path in paths:
                if os.path.exists(path):
                    docs = PyPDFLoader(path).load()
                    docs = self.text_splitter.split_documents(docs)
                    for d in docs:
                        d.metadata["role"] = role
                        d.metadata["source"] = path
                    chunks.extend(docs)
            self.collections[role] = chunks

    def create_chroma_collections(self):
        for role, docs in self.collections.items():
            if not docs:
                print(f"⚠️ No docs to create Chroma collection for role: {role}")
                continue
            role_dir = os.path.join(self.persist_directory, role)
            os.makedirs(role_dir, exist_ok=True)
            self.chroma_collections[role] = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=role_dir,
                collection_name=f"{role}_collection"
            )

    def get_chroma_collection(self, role):
        return self.chroma_collections.get(role)

    def similarity_search_across_roles(self, query: str, roles: List[str], k: int = 3):
        results = []
        for role in roles:
            chroma = self.get_chroma_collection(role)
            if chroma:
                results.extend(chroma.similarity_search(query, k=k))
        return results

    def get_context(self, role, query, k=3):
        accessible = ROLE_HIERARCHY.get(role, [])
        docs = self.similarity_search_across_roles(query, accessible, k=k)
        if not docs:
            return "No relevant documents found."
        context = ""
        for i, d in enumerate(docs):
            source = os.path.basename(d.metadata.get("source", "Unknown"))
            context += f"\nDoc {i+1} ({source}): {d.page_content.strip()}\n"
        return context

# ─────────────────────── Singleton Chunker Init ───────────────── #

chunker = RoleBasedPDFChunker(ROLE_SOURCES)
chunker.load_and_chunk_pdfs()

print("✅ Loaded all PDFs and split chunks.")

chunker.create_chroma_collections()
print("✅ Created Chroma collections.")

for role in ROLE_SOURCES:
    role_dir = os.path.join("chroma_db", role)
    if os.path.exists(role_dir):
        print(f"✅ Collection directory created: {role_dir}")
    else:
        print(f"❌ Missing collection directory: {role_dir}")


# ─────────────────────── API Endpoints ────────────────────────── #

@app.get("/")
def read_root():
    return {"status":"ok", "message":"Fastapi is live"}


@app.post("/login")
def login(payload: LoginRequest):
    for user in users:
        if user["name"] == payload.name and user["password"] == payload.password:
            return {"uuid": user["uuid"], "username": user["full_name"]}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/ask")
async def ask(payload: AskRequest, user=Depends(validate_uuid)):
    user_uuid = user["uuid"]
    user_role = user["role"]
    query = payload.query

    # Use in-session memory, initialize if not present
    if user_uuid not in session_memory_store:
        session_memory_store[user_uuid] = ListMemory()
    memory = session_memory_store[user_uuid]


    # Context from ChromaDB
    context = chunker.get_context(user_role, query)

    # Construct system prompt
    SYSTEM_PROMPT = f"""
    You are a document-driven assistant for the {user_role} role. 
    Answer the user's question ONLY based on the following document excerpts:

    {context}

    - Answer user queries EXCLUSIVELY by citing and paraphrasing those document excerpts.
    - DO NOT use any external knowledge or guesswork.
    - Be concise, neutral, and professional.
    - Do not speculate or provide suggestions unless clearly stated in the documents.
    - If the answer cannot be found in the documents, respond:
    "I'm sorry, I couldn't find that information in the provided documents."
    """

    # Assistant with session memory
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    assistant = AssistantAgent(
        name="rag_assistant",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        memory=[memory]
    )

    # Add query to memory
    await memory.add(MemoryContent(content=f"User: {query}", mime_type=MemoryMimeType.TEXT))

    # Run assistant
    cancellation_token = CancellationToken()
    result = await assistant.on_messages([TextMessage(content=query, source="user")], cancellation_token)

    response_text = result.chat_message.content if result else "No response generated."

    await memory.add(MemoryContent(content=f"Assistant: {response_text}", mime_type=MemoryMimeType.TEXT))


    return {"message": response_text}

