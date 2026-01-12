from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import os
import hashlib
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sentence_transformers import SentenceTransformer

# -------------------------
# ENV + Streamlit config
# -------------------------
load_dotenv()
st.set_page_config(page_title="Çoklu Model Doküman Asistanı (LangGraph)", layout="wide")

DEFAULT_PDF = "one_piece.pdf"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

has_gemini = bool(GOOGLE_API_KEY)
has_openai = bool(OPENAI_API_KEY)


# -------------------------
# Helpers
# -------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_uploaded_pdf(uploaded_file) -> tuple[str, str]:
    data = uploaded_file.getbuffer().tobytes()
    h = sha256_bytes(data)
    path = f"/tmp/{h}_{uploaded_file.name}"
    # Windows kullanıyorsan /tmp yerine os.getcwd() kullanmak daha güvenlidir, 
    # ama şimdilik böyle bırakıyorum.
    with open(path, "wb") as f:
        f.write(data)
    return path, h

def load_default_pdf() -> tuple[str, str]:
    if not os.path.exists(DEFAULT_PDF):
        return "", ""
    with open(DEFAULT_PDF, "rb") as f:
        data = f.read()
    return DEFAULT_PDF, sha256_bytes(data)

def format_docs_for_prompt(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        prefix = f"[Sayfa {page+1}] " if isinstance(page, int) else ""
        parts.append(prefix + d.page_content)
    return "\n\n".join(parts)

def show_sources_ui(docs, max_chars: int = 350):
    with st.expander("📚 Kullanılan Kaynak Parçalar"):
        for i, d in enumerate(docs, start=1):
            page = d.metadata.get("page")
            page_str = f"Sayfa {page+1}" if isinstance(page, int) else "Sayfa ?"
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "…"
            st.markdown(f"**{i}. {page_str}** — {snippet}")


# -------------------------
# Local Embeddings (Sınırsız & Ücretsiz)
# -------------------------
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # PDF parçalarını vektöre çevirir
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text):
        # Sorunu vektöre çevirir
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return v.tolist()

@st.cache_resource
def get_embeddings():
    # Tekrar yerel modele dönüyoruz
    return LocalSentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Ayarlar")

model_secimi = st.sidebar.radio(
    "Cevap Modeli:",
    ("Gemini 3 Flash Preview", "OpenAI GPT-3.5 Turbo")
)

show_sources = st.sidebar.toggle("Kaynakları göster", value=True)
uploaded_file = st.sidebar.file_uploader("PDF yükle (opsiyonel)", type=["pdf"])

pdf_path = ""
pdf_hash = ""

if uploaded_file is not None:
    pdf_path, pdf_hash = save_uploaded_pdf(uploaded_file)
else:
    pdf_path, pdf_hash = load_default_pdf()
    if not pdf_path:
        st.sidebar.warning(f"Varsayılan PDF ({DEFAULT_PDF}) bulunamadı. Lütfen PDF yükleyin.")
        st.stop()


# -------------------------
# Build Retriever (cache)
# -------------------------
@st.cache_resource
def build_retriever(pdf_path: str, pdf_hash: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = get_embeddings() #isim değişti
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = build_retriever(pdf_path, pdf_hash)


# -------------------------
# UI header & LLM Selection
# -------------------------
st.title("One Piece Assistant 🏴‍☠️👒🍖 ")
st.caption(f"Aktif Doküman: {os.path.basename(pdf_path)} | Seçili Model: {model_secimi}")

llm = None
if model_secimi == "Gemini 3 Flash Preview":
    if not has_gemini:
        st.error("Gemini seçili ama GOOGLE_API_KEY yok.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
else:
    if not has_openai:
        st.error("OpenAI seçili ama OPENAI_API_KEY yok.")
        st.stop()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ==========================================
# LANGGRAPH YAPISI (YENİ EKLENEN KISIM)
# ==========================================

# 1. State (Durum) Tanımlaması
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# 2. Node (Düğüm) Fonksiyonları

def retrieve(state):
    """
    Belgeleri veritabanından çeker.
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Belgeleri kullanarak cevabı üretir.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    if llm is None:
        return {"generation": "LLM yapılandırılmamış."}
    
    # Prompt Tanımı
    system_prompt = (
        "Sen 'One Piece Assistant' adında yardımcı bir asistansın.\n"
        "Görevlerin:\n"
        "1. Sohbet sorularına nazikçe cevap ver.\n"
        "2. Bilgi sorularını SADECE aşağıdaki bağlamı kullanarak cevapla.\n"
        "3. Bilgi yoksa 'Bilmiyorum' de.\n\n"
        "BAĞLAM:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    
    # Zinciri oluştur
    rag_chain = prompt | llm | StrOutputParser()
    
    # Context formatla ve çalıştır
    context_text = format_docs_for_prompt(documents)
    generation = rag_chain.invoke({"context": context_text, "input": question})
    
    return {"generation": generation}

# 3. Grafiği İnşa Etme
workflow = StateGraph(GraphState)

# Düğümleri ekle
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Kenarları (Bağlantıları) ekle
workflow.set_entry_point("retrieve") # Başlangıç noktası
workflow.add_edge("retrieve", "generate") # Retrieve bitince Generate'e git
workflow.add_edge("generate", END) # Generate bitince Bitir

# Grafiği derle
app_graph = workflow.compile()


# -------------------------
# Chat UI
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Sorunuzu yazın...")

if user_input:
    # Kullanıcı mesajını ekrana yaz
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # --- LANGGRAPH ÇALIŞTIRMA KISMI ---
            inputs: GraphState = {
                "question": user_input,
                "documents": [],  # Başlangıçta boş, retrieve düğümü dolduracak
                "generation": ""  # Başlangıçta boş, generate düğümü dolduracak
            }

            # invoke ile grafiği çalıştırıyoruz
            result = app_graph.invoke(inputs)
            
            # Sonuçları al
            answer = result["generation"]
            source_docs = result["documents"]
            
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            if show_sources:
                show_sources_ui(source_docs)

        except Exception as e:
            st.error(f"Hata: {e}")