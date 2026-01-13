import os
import json
import uuid
import glob
import random  
from datetime import datetime
from typing import TypedDict, List
from dotenv import load_dotenv
import streamlit as st

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# -------------------------
# 1. TEMEL AYARLAR VE "RUH" LİSTELERİ
# -------------------------
load_dotenv()
st.set_page_config(
    page_title="Going-Chaty 𓊝",
    page_icon="🏴‍☠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- RASTGELE EĞLENCE LİSTELERİ ---
PIRATE_GREETINGS = [
    "Yohoho! Hoş geldin Kaptan! 💀",
    "Kaizoku ou ni ore wa naru! (Korsanlar Kralı olacağım!) 🍖",
    "Oii, Luffy! Bugün nereye yelken açıyoruz? 🌊",
    "Selamlar! Bugün deniz çok güzel, değil mi? ☀️",
    "Tayfa hazır Kaptan, emirlerini bekliyoruz! 🏴‍☠️",
    "Bugün macera kokusu alıyorum! 👃🍖"
]

LOADING_MESSAGES = [
    "Log Pose ayarlanıyor... 🧭",
    "Denizcilerden kaçıyorum... ⚓",
    "Sanji mutfakta bir şeyler hazırlıyor... 🍳",
    "Zoro yine kayboldu, onu arıyorum... ⚔️",
    "Franky gemiyi tamir ediyor... 🔨",
    "Nami haritaları kontrol ediyor... 🗺️",
    "Robin antik metinleri okuyor... 📚",
    "Usopp yalan... öhm, kahramanlık hikayesi anlatıyor... 🤥",
    "Brook bir şarkı mırıldanıyor (gerçi dudağı yok ama)... 💀🎶",
    "Deniz Canavarları ile boğuşuyorum... 🦑"
]

# -------------------------
# 2. PREMIUM CSS TASARIMI
# -------------------------
st.markdown("""
<style>
    /* Font ve Genel Stil */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1 { color: #FF4B4B; font-weight: 700; }

    /* --- İKONLARI YOK ETME (AVATAR) --- */
    [data-testid="stChatMessageAvatarContainer"] {
        display: none !important;
    }
    [data-testid="stChatMessageContent"] {
        margin-left: 0px !important;
        padding-left: 0px !important;
    }
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0px !important;
        margin-top: 5px !important;
        margin-bottom: 5px !important;
    }

    /* --- BALONCUK TASARIMLARI --- */
    .user-container {
        display: flex;
        justify-content: flex-end;
        width: 100%;
    }
    .user-bubble {
        background-color: #FF4B4B; 
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 4px 20px;
        max-width: 80%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.5;
    }
    
    .bot-container {
        display: flex;
        justify-content: flex-start;
        width: 100%;
    }
    .bot-bubble {
        background-color: #f4f6f9; 
        color: #2c3e50;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 4px;
        max-width: 80%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 15px;
        line-height: 1.5;
        border: 1px solid #e1e4e8;
    }
    
    @media (prefers-color-scheme: dark) {
        .bot-bubble {
            background-color: #262730;
            color: #ececec;
            border: 1px solid #444;
        }
    }

    /* --- SIDEBAR TASARIMI (KART GÖRÜNÜMÜ) --- */
    section[data-testid="stSidebar"] button {
        border-radius: 10px !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease;
        margin-bottom: 5px;
    }

    button[kind="secondary"] {
        background-color: white; 
        text-align: left;
        padding-left: 15px;
        font-weight: 500;
        color: #444;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    @media (prefers-color-scheme: dark) {
        button[kind="secondary"] {
            background-color: #262730;
            color: #ddd;
            border: 1px solid #444 !important;
        }
    }

    button[kind="secondary"]:hover {
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
        transform: translateX(3px); 
    }

    button[help="Ayarlar"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        font-size: 1.2rem;
        color: #888;
    }
    button[help="Ayarlar"]:hover {
        color: #FF4B4B !important;
        transform: none !important;
    }

</style>
""", unsafe_allow_html=True)

DEFAULT_PDF = "one_piece.pdf"
HISTORY_FOLDER = "chat_history"

if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
has_gemini = bool(GOOGLE_API_KEY)
has_openai = bool(OPENAI_API_KEY)


# -------------------------
# 3. YARDIMCI FONKSİYONLAR
# -------------------------
def save_chat(session_id, messages, title=None):
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    current_data = {"title": None, "messages": []}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    current_data["messages"] = loaded
                else:
                    current_data = loaded
        except:
            pass

    current_data["messages"] = messages
    if title:
        current_data["title"] = title
    elif not current_data["title"] and messages:
        first_msg = next((m["content"] for m in messages if m["role"] == "user"), "Yeni Sohbet")
        current_data["title"] = (first_msg[:25] + '..') if len(first_msg) > 25 else first_msg

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

def load_chat(session_id):
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {"title": "Geçmiş Sohbet", "messages": data}
                return data
        except:
            pass
    return {"title": "Yeni Sohbet", "messages": []}

def rename_chat(session_id, new_title):
    data = load_chat(session_id)
    data["title"] = new_title
    save_chat(session_id, data["messages"], new_title)

def delete_chat(session_id):
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def get_all_chats():
    files = glob.glob(os.path.join(HISTORY_FOLDER, "*.json"))
    chats = []
    for f in files:
        filename = os.path.basename(f).replace(".json", "")
        timestamp = os.path.getctime(f)
        date_str = datetime.fromtimestamp(timestamp).strftime('%d.%m %H:%M')
        
        title = "Yeni Sohbet"
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict) and "title" in data:
                    title = data["title"]
                elif isinstance(data, list):
                    first_msg = next((m["content"] for m in data if m["role"] == "user"), "Yeni Sohbet")
                    title = (first_msg[:20] + '..') if len(first_msg) > 20 else first_msg
        except:
            pass
        chats.append({"id": filename, "date": date_str, "title": title})
    chats.sort(key=lambda x: x["date"], reverse=True)
    return chats

def format_docs_for_prompt(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"[Sayfa {page}] {d.page_content}")
    return "\n\n".join(parts)

def format_history_for_prompt(messages) -> str:
    formatted = ""
    for msg in messages:
        role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

# -------------------------
# 4. EMBEDDINGS & RETRIEVER
# -------------------------
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]
    def embed_query(self, text):
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return v.tolist()

@st.cache_resource
def get_embeddings():
    return LocalSentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_retriever(pdf_path: str):
    if not os.path.exists(pdf_path): return None
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = build_retriever(DEFAULT_PDF)


# -------------------------
# 5. MENÜ VE SIDEBAR (MODAL/DIALOG İLE)
# -------------------------

@st.dialog("🏴‍☠️ Sohbet Seçenekleri")
def options_menu(chat_id, current_title):
    st.write(f"Düzenlenen: **{current_title}**")
    new_name = st.text_input("Yeni İsim:", value=current_title)
    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("💾 Kaydet", type="primary", use_container_width=True):
            rename_chat(chat_id, new_name)
            st.rerun()
    with col_del:
        if st.button("🗑️ Sil", type="secondary", use_container_width=True):
            delete_chat(chat_id)
            if st.session_state.session_id == chat_id:
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
            st.rerun()

# --- Sidebar Yapısı ---
st.sidebar.title("Going-Chaty")

# Anlık Durum (Ruh Katma Kısmı)
locations = ["Wano Ülkesi", "Egghead Adası", "Thousand Sunny", "Denizci Üssü G-5", "Elbaf"]
current_loc = random.choice(locations)
st.sidebar.caption(f"📍 Konum: {current_loc}")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

if st.sidebar.button("🧭 Yeni Macera (Sohbet)", type="primary"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")

previous_chats = get_all_chats()
if not previous_chats:
    st.sidebar.info("Henüz macera günlüğü boş.")

for chat in previous_chats:
    col1, col2 = st.sidebar.columns([0.85, 0.15])
    with col1:
        if st.button(chat['title'], key=f"chat_{chat['id']}", use_container_width=True, type="secondary"):
            st.session_state.session_id = chat['id']
            loaded_data = load_chat(chat['id'])
            st.session_state.messages = loaded_data["messages"]
            st.rerun()
    with col2:
        if st.button("⋮", key=f"opt_{chat['id']}", help="Ayarlar"):
            options_menu(chat['id'], chat['title'])

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️Gemi Ayarları"):
    model_secimi = st.radio("Model:", ("Gemini 3 Flash Preview", "OpenAI GPT-3.5 Turbo"))
    show_sources = st.toggle("Kaynakları göster", value=True)

llm = None
if model_secimi == "Gemini 3 Flash Preview":
    if not has_gemini: st.error("Gemini Key Yok!"); st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
else:
    if not has_openai: st.error("OpenAI Key Yok!"); st.stop()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# -------------------------
# 6. LANGGRAPH
# -------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    chat_history: str

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question) if retriever else []
    return {"documents": documents, "question": question}

def generate(state):
    print("---GENERATE---")
    if llm is None: return {"generation": "Hata: Model yok."}
    
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    
    # RUH KATILMIŞ PROMPT 🏴‍☠️
    system_prompt = (
        "Sen 'Going-Chaty' adında, Hasır Şapka ruhuna sahip, neşeli ve sadık bir asistansın. "
        "Kullanıcıya 'Kaptan' veya 'Nakama' diye hitap edebilirsin.\n\n"
        "GÖREVLERİN:\n"
        "1. **Sohbet:** Eğer konu geyik, muhabbet veya senin fikrinse; yaratıcı ol! One Piece terimleri kullan (Haki, Berry, Denizciler vb.). "
        "Gülüş efektleri kullanmaktan çekinme (Örn: Shishishi, Yohohoho, Zehahaha).\n"
        "2. **Bilgi:** Eğer Kaptan (kullanıcı) teknik veya PDF ile ilgili bir şey sorarsa, "
        "ciddileş ve SADECE aşağıdaki BAĞLAM (Context) bilgisini kullanarak net bir cevap ver.\n"
        "3. **Bilinmeyen:** Bilgi bağlamda yoksa, 'Kaptan, bu bilgi seyir defterimde (PDF) yok ama seninle teoriler üzerine konuşabilirim!' de.\n\n"
        "SOHBET GEÇMİŞİ:\n{chat_history}\n\n"
        "BAĞLAM:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    rag_chain = prompt | llm | StrOutputParser()
    context_text = format_docs_for_prompt(documents)
    generation = rag_chain.invoke({"context": context_text, "chat_history": chat_history, "input": question})
    return {"generation": generation}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# -------------------------
# 7. ARAYÜZ (CHAT UI)
# -------------------------
# Rastgele Karşılama Mesajı (Sadece sayfa ilk yüklendiğinde ve mesaj yoksa)
if not st.session_state.messages:
    welcome_msg = random.choice(PIRATE_GREETINGS)
    st.title(f"Going-Chaty 👒 {welcome_msg.split('!')[0]}!")
else:
    st.title("Going-Chaty 👒🍖")

# Mesajları Göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=None):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-container"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-container"><div class="bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

# Input
user_input = st.chat_input("Grand Line'da bir soru sor...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=None):
        st.markdown(f'<div class="user-container"><div class="user-bubble">{user_input}</div></div>', unsafe_allow_html=True)
    
    history_text = format_history_for_prompt(st.session_state.messages[:-1])
    
    with st.chat_message("assistant", avatar=None):
        placeholder = st.empty()
        inputs: GraphState = {"question": user_input, "documents": [], "generation": "", "chat_history": history_text}
        
        # RUH: Rastgele Bekleme Mesajı
        random_loader = random.choice(LOADING_MESSAGES)
        with st.spinner(random_loader):
            result = app_graph.invoke(inputs)
        
        answer = result["generation"]
        source_docs = result["documents"]
        
        placeholder.markdown(f'<div class="bot-container"><div class="bot-bubble">{answer}</div></div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        current_data = load_chat(st.session_state.session_id)
        current_title = current_data.get("title")
        save_chat(st.session_state.session_id, st.session_state.messages, title=current_title)

        if show_sources and source_docs:
            with st.expander("𓂃🪶Seyir Defteri Kayıtları"):
                for i, d in enumerate(source_docs, 1):
                    st.markdown(f"**{i}.** {d.page_content[:200]}...")