import os
import json
import uuid
import glob
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
# 1. TEMEL AYARLAR
# -------------------------
load_dotenv()
st.set_page_config(
    page_title="⊹ ࣪ ﹏𓊝﹏𓂁﹏⊹ ࣪ ˖",
    page_icon="🏴‍☠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Özel Tasarım (Gemini/ChatGPT Tarzı)
# -------------------------
st.markdown("""
<style>
    /* Ana başlık rengi */
    h1 { color: #FF4B4B; }
    
    /* Chat mesajları çerçevesi */
    .stChatMessage {
        border: 1px solid #333; /* Hafif çerçeve */
        border-radius: 12px;
        padding: 15px;
    }

    /* --- SIDEBAR TASARIMI --- */
    
    /* Sidebar'daki "Secondary" butonları (Geçmiş sohbetler) şeffaf yap */
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background-color: transparent;
        border: none;
        text-align: left; /* Yazıyı sola yasla */
        width: 100%;
        color: inherit; /* Temaya uygun renk */
        padding: 10px;
        transition: all 0.2s ease; /* Yumuşak geçiş */
    }

    /* Üzerine gelince (Hover) hafif gri/beyaz olsun */
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.1); /* Hafif aydınlatma */
        padding-left: 15px; /* Hafif sağa kayma efekti */
        color: #FF4B4B;
    }

    /* "Yeni Sohbet" butonu (Primary) dikkat çekici kalsın */
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Sidebar başlıklarını biraz küçültelim */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_PDF = "one_piece.pdf"
HISTORY_FOLDER = "chat_history"

# Klasör yoksa oluştur
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
has_gemini = bool(GOOGLE_API_KEY)
has_openai = bool(OPENAI_API_KEY)

# -------------------------
# 2. YARDIMCI FONKSİYONLAR (Storage & PDF)
# -------------------------
def save_chat_history(session_id, messages):
    """Sohbeti JSON olarak kaydeder"""
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def load_chat_history(session_id):
    """JSON'dan sohbeti yükler"""
    filepath = os.path.join(HISTORY_FOLDER, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_all_chats():
    """Tüm kayıtlı sohbetleri listeler ve ilk mesajı başlık yapar"""
    files = glob.glob(os.path.join(HISTORY_FOLDER, "*.json"))
    chats = []
    for f in files:
        filename = os.path.basename(f).replace(".json", "")
        timestamp = os.path.getctime(f)
        date_str = datetime.fromtimestamp(timestamp).strftime('%d.%m %H:%M')
        
        # Dosyanın içini oku ve ilk mesajı al
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                # İlk kullanıcı mesajını bul
                first_msg = next((m["content"] for m in data if m["role"] == "user"), "Yeni Sohbet")
                # Çok uzunsa kısalt (30 karakter)
                title = (first_msg[:25] + '..') if len(first_msg) > 25 else first_msg
        except:
            title = "Yeni Sohbet"

        chats.append({"id": filename, "date": date_str, "title": title})
    
    # En yeniden eskiye sırala
    chats.sort(key=lambda x: x["date"], reverse=True)
    return chats

def format_docs_for_prompt(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"[Sayfa {page}] {d.page_content}")
    return "\n\n".join(parts)

def format_history_for_prompt(messages) -> str:
    """Mesaj listesini LLM'in anlayacağı metne çevirir"""
    formatted = ""
    for msg in messages:
        role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

# -------------------------
# 3. EMBEDDINGS & RETRIEVER (Local)
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
    if not os.path.exists(pdf_path):
        return None
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Retriever'ı başlat
retriever = build_retriever(DEFAULT_PDF)

# -------------------------
# 4. SIDEBAR & SESSION MANAGEMENT
# -------------------------
st.sidebar.title("🗂️ Sohbet Geçmişi")

# Session ID Kontrolü
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

# Yeni Sohbet Butonu
if st.sidebar.button("﹏𓊝﹏ Yeni Sohbet Başlat", type="primary"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()

# Eski Sohbetleri Listele
previous_chats = get_all_chats()

# Eğer hiç sohbet yoksa bilgi ver
if not previous_chats:
    st.sidebar.caption("Henüz geçmiş sohbet yok.")

for chat in previous_chats:
    # Butonun üzerinde artık "Luffy kimdir?" gibi başlık yazacak
    # Altına da küçük tarih ekliyoruz
    label = f"{chat['title']}" 
    
    # kind="secondary" diyerek CSS'in bunu yakalamasını sağlıyoruz
    if st.sidebar.button(label, key=chat['id'], use_container_width=True, type="secondary"):
        st.session_state.session_id = chat['id']
        st.session_state.messages = load_chat_history(chat['id'])
        st.rerun()

# Ayarlar
st.sidebar.divider()
st.sidebar.subheader("⚙️ Ayarlar")
model_secimi = st.sidebar.radio("Model:", ("Gemini 3 Flash Preview", "OpenAI GPT-3.5 Turbo"))
show_sources = st.sidebar.toggle("Kaynakları göster", value=True)

# LLM Seçimi
llm = None
if model_secimi == "Gemini 3 Flash Preview":
    if not has_gemini:
        st.error("Gemini API Key eksik!")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
else:
    if not has_openai:
        st.error("OpenAI API Key eksik!")
        st.stop()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# -------------------------
# 5. LANGGRAPH (MEMORY DESTEKLİ)
# -------------------------

# State Tanımı (Artık history de taşıyor)
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    chat_history: str  # <--- YENİ: Geçmiş sohbet metni

def retrieve(state):
    """Belgeleri bulur"""
    print("---RETRIEVE---")
    question = state["question"]
    if retriever:
        documents = retriever.invoke(question)
    else:
        documents = []
    return {"documents": documents, "question": question}

def generate(state):
    """Cevabı üretir"""
    print("---GENERATE---")

    if llm is None:
        return {"generation": "Hata: Bir yapay zeka modeli seçilmedi veya API anahtarı eksik."}
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"] # Geçmişi al
    
    # Prompt - Artık hafızası var!
    system_prompt = (
        "Sen 'Going-Chaty' One Piece evrenine hakim, neşeli ve yardımsever bir asistansın.\n"
        "Görevlerin şunlar:\n\n"
        "1. **SOHBET VE YORUM:** Eğer kullanıcı senin fikrini sorarsa (Örn: 'Hangi meyveyi istersin?', 'En sevdiğin karakter kim?'), "
        "bağlama bağlı kalmak zorunda değilsin :). Yaratıcı, eğlenceli ve bir One Piece hayranı gibi cevap ver. "
        "(Örn: 'Gomu Gomu no Mi isterdim çünkü uçmak çok havalı!' gibi).\n\n"
        "2. **BİLGİ SORULARI:** Eğer kullanıcı dokümanla ilgili teknik veya bilgi içerikli bir soru sorarsa, "
        "cevabı SADECE aşağıdaki BAĞLAM (Context) bilgisini kullanarak ver.\n\n"
        "3. **BİLİNMEYEN BİLGİ:** Eğer sorulan *bilgi* bağlamda yoksa dürüstçe 'Bu detay dokümanlarda geçmiyor ama istersen seninle teoriler üzerine konuşabiliriz!' de.\n\n"
        "SOHBET GEÇMİŞİ:\n{chat_history}\n\n"
        "BAĞLAM:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    context_text = format_docs_for_prompt(documents)
    
    # Zinciri çalıştır
    generation = rag_chain.invoke({
        "context": context_text, 
        "chat_history": chat_history, 
        "input": question
    })
    
    return {"generation": generation}

# Graph Oluşturma
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# -------------------------
# 6. ARAYÜZ (CHAT UI)
# -------------------------
st.title("Going-Chaty 👒🍖🏴‍☠️🍈☀️")

# Mesajları Ekrana Yaz
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Kullanıcı Girdisi
user_input = st.chat_input("Sorunuzu yazın...")

if user_input:
    # 1. Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # 2. Geçmişi metne çevir (LLM için)
    # Son mesajı hariç tutuyoruz ki tekrar etmesin (zaten input olarak gidiyor)
    history_text = format_history_for_prompt(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        try:
            inputs: GraphState = {
                "question": user_input,
                "documents": [],
                "generation": "",
                "chat_history": history_text # <--- Geçmişi gönderiyoruz
            }
            
            result = app_graph.invoke(inputs)
            
            answer = result["generation"]
            source_docs = result["documents"]
            
            st.write(answer)
            
            # 3. Cevabı kaydet
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # 4. Dosyaya Kalıcı Olarak Kaydet (JSON)
            save_chat_history(st.session_state.session_id, st.session_state.messages)

            if show_sources and source_docs:
                with st.expander("📚 Kaynaklar"):
                    for i, d in enumerate(source_docs, 1):
                        st.markdown(f"**{i}.** {d.page_content[:200]}...")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")