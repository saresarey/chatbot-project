import os
import hashlib
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from sentence_transformers import SentenceTransformer


# -------------------------
# ENV + Streamlit config
# -------------------------
load_dotenv()
st.set_page_config(page_title="Çoklu Model Doküman Asistanı (Hybrid)", layout="wide")

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
# Local Embeddings (FREE)
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
def get_local_embeddings():
    # İstersen daha iyi (ama daha ağır) model:
    # "sentence-transformers/all-mpnet-base-v2"
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


if model_secimi == "Gemini 3 Flash Preview":
    if not has_gemini:
        st.error("Gemini seçili ama GOOGLE_API_KEY yok. Lütfen .env ekleyin.")
        st.stop()
    # Düzelttiğimiz satır 
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

# -------------------------
# Build Retriever (cache)
# -------------------------
@st.cache_resource
def build_retriever(pdf_path: str, pdf_hash: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = get_local_embeddings()  # ✅ LOCAL / FREE
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})
retriever = build_retriever(pdf_path, pdf_hash)


# -------------------------
# UI header
# -------------------------
st.title("One Piece Assistant")
st.caption(f"Aktif Doküman: {os.path.basename(pdf_path)} | Seçili Model: {model_secimi}")

# LLM seç
llm = None
if model_secimi == "Gemini 3 Flash Preview":
    if not has_gemini:
        st.error("Gemini seçili ama GOOGLE_API_KEY yok. Lütfen .env ekleyin.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
else:
    if not has_openai:
        st.error("OpenAI seçili ama OPENAI_API_KEY yok. Lütfen .env ekleyin.")
        st.stop()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# -------------------------
# Prompt + Runnable RAG
# -------------------------
# -------------------------
# Prompt + Runnable RAG
# -------------------------
# -------------------------
# Prompt + Runnable RAG
# -------------------------
system_prompt = (
    "Sen 'One Piece Assistant' adında yardımcı ve samimi bir asistansın.\n"
    "Görevlerin şunlar:\n"
    "1. Eğer kullanıcı seninle sohbet ederse (Merhaba, Nasılsın, Günaydın vb.) veya vedalaşırsa, onlara nazikçe ve samimi bir dille karşılık ver.\n"
    "2. Eğer kullanıcı bir bilgi sorarsa, cevabı SADECE aşağıdaki bağlamı (context) kullanarak ver.\n"
    "3. Eğer sorulan bilgi bağlamda yoksa, dürüstçe 'Bu bilgi dokümanda yer almıyor.' de, asla uydurma.\n\n"
    "BAĞLAM:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

retriever_runnable = RunnableLambda(lambda q: retriever.invoke(q))

rag_chain = (
    {
        "context": retriever_runnable | RunnableLambda(format_docs_for_prompt),
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
# -------------------------
# Chat UI
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Sorunuzu yazın...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            docs = retriever.invoke(user_input)
            answer = rag_chain.invoke(user_input)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            if show_sources:
                show_sources_ui(docs)

        except Exception as e:
            st.error(f"Hata: {e}")
            # OpenAI 429 gibi durumlarda kullanıcıya daha net anlat:
            msg = str(e)
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                st.info("OpenAI API kotanız/billing yok. Ödev için seçenek dursun ama çalışması için billing gerekir. "
                        "Gemini'yi seçebilir veya OpenAI için faturalandırma açabilirsiniz.")
