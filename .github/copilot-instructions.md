# Copilot Project Instructions

- **App shape**: Single Streamlit app in [app.py](app.py) implementing a simple RAG chatbot over one PDF; no backend services beyond Streamlit session state and in-memory Chroma.
- **Data flow**: PDF `PDF_DOSYA_ADI` loads via `PyPDFLoader`, split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, embedded with Google Generative AI embeddings (`models/embedding-001`), stored in a transient `Chroma.from_documents` vectorstore, exposed as a retriever cached with `@st.cache_resource`.
- **LLM choice**: Sidebar radio selects either `ChatGoogleGenerativeAI` (`gemini-1.5-flash`, temperature 0) or `ChatOpenAI` (`gpt-3.5-turbo`, temperature 0). If OpenAI key missing, selection auto-falls back to Gemini with a sidebar warning.
- **Prompting**: RAG chain built with `create_stuff_documents_chain` + `create_retrieval_chain`; system prompt instructs to answer only from context and reply "Bilmiyorum" when context insufficient.
- **UI state**: Chat history kept in `st.session_state.messages`; each user turn triggers `rag_chain.invoke({"input": user_input})` and appends `response["answer"]`.
- **Environment**: Requires `.env` with `GOOGLE_API_KEY` (mandatory) and optionally `OPENAI_API_KEY`; app halts early if Google key missing. No other config files.
- **Running**: From project root, install deps (`pip install -r requirements.txt`) and start with `streamlit run app.py`. Current requirements listed in [requirements.txt](requirements.txt).
- **Documents**: Expected PDF filename set in `PDF_DOSYA_ADI` near top of [app.py](app.py#L8); ensure the file exists in the repo root before running. Chroma store is ephemeral per run (no persistence path configured).
- **Error handling**: Loader or missing PDF errors surface via `st.error` and stop setup. Chat invocation wrapped in try/except to display errors inline without crashing the session.
- **Style & conventions**: Keep Turkish UI text and emojis as-is; favor concise Streamlit components; avoid removing `@st.cache_resource` unless changing caching behavior intentionally.
- **Extending**: When adding features, respect the single-page Streamlit flow; use the existing retriever/LLM selection pattern and session message handling instead of introducing new global state managers.
- **Testing/debugging**: No test suite present; quickest manual check is running `streamlit run app.py` and verifying sidebar model toggle, PDF load, and chat responses sourced from the document.

If any part of this guide is unclear or incomplete, tell me what to refine and I will update it.