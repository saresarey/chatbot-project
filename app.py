import streamlit as st
import os
from dotenv import load_dotenv

# --- 1. AYARLAR ---
st.set_page_config(page_title="Çoklu Model Asistanı", layout="wide")
load_dotenv()

# PDF Dosya Adını Buraya Yaz
PDF_DOSYA_ADI = "one_piece.pdf"

# API Anahtarlarını Kontrol Et
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ HATA: GOOGLE_API_KEY eksik!")
    st.stop()
    
# OpenAI anahtarı yoksa sadece uyarı verelim, programı durdurmayalım (Gemini çalışsın diye)
has_openai = os.getenv("OPENAI_API_KEY") is not None

# --- 2. KÜTÜPHANELER ---
# try-except bloğunu kaldırdık ki GERÇEK hatayı görelim
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --- 3. YAN MENÜ (MODEL SEÇİMİ) ---
st.sidebar.title("⚙️ Ayarlar")
st.sidebar.markdown("Cevap verecek yapay zeka modelini seç:")

model_secimi = st.sidebar.radio(
    "Model:",
    ("Google Gemini 1.5 Flash", "OpenAI GPT-3.5 Turbo")
)

# Eğer OpenAI seçildiyse ama anahtar yoksa uyar
if model_secimi == "OpenAI GPT-3.5 Turbo" and not has_openai:
    st.sidebar.error("⚠️ OpenAI API Anahtarı bulunamadı! Gemini'ye geçiliyor.")
    model_secimi = "Google Gemini 1.5 Flash"

# --- 4. RAG SİSTEMİ ---
@st.cache_resource
def setup_rag_system():
    if not os.path.exists(PDF_DOSYA_ADI):
        st.error(f"❌ '{PDF_DOSYA_ADI}' dosyası bulunamadı!")
        return None

    # A) PDF Yükle ve Parçala
    loader = PyPDFLoader(PDF_DOSYA_ADI)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # B) Embedding (Google kullanmaya devam ediyoruz, ücretsiz ve hızlı)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

# Retriever'ı bir kez kur (Maliyetten tasarruf için)
retriever = setup_rag_system()

if retriever:
    st.title("🤖 Çoklu Model Doküman Asistanı")
    st.caption(f"Aktif Doküman: {PDF_DOSYA_ADI} | Seçili Model: {model_secimi}")

    # --- 5. MODELİ AYARLA (Dinamik Kısım) ---
    if model_secimi == "Google Gemini 1.5 Flash":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    else:
        # OpenAI Modelini Başlat
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt (Talimat)
    system_prompt = (
        "Sen yardımcı bir asistansın. Soruları SADECE aşağıdaki bağlamı kullanarak cevapla. "
        "Bilmiyorsan 'Bilmiyorum' de. "
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Zinciri Oluştur
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # --- 6. SOHBET ARAYÜZÜ ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Hata: {e}")