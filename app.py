import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

GOOGLE_API_KEY = "AIzaSyAirGKFXwyaMurrXF1K8ZAFcUwoUN7LqyE"


def toggle_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as accurately and thoroughly as possible based on the provided context.
    If the answer is not available in the context, respond with: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index not found. Please upload and process PDFs first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.markdown("### 😊 Reply:")
    st.success(response["output_text"])

def main():
    st.set_page_config(page_title="PDF Reader", page_icon="📑", layout="wide")


    if "theme" not in st.session_state:
        st.session_state.theme = "light"

 
    if st.session_state.theme == "light":
        background_color = "#e0eafc"
        text_color = "#333"
        button_color = "#1e3a8a"
        sidebar_color = "#1e3a8a"
    else:
        background_color = "#121212"
        text_color = "#ffffff"
        button_color = "#bb86fc"
        sidebar_color = "#333333"

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background_color}; font-family: 'Arial', sans-serif; color: {text_color}; }}
        .sidebar .sidebar-content {{ background-color: {sidebar_color}; color: white; padding: 20px; border-radius: 15px; }}
        .css-1d391kg {{ background: rgba(255, 255, 255, 0.7); border-radius: 15px; padding: 25px; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); }}
        .stTextInput>div>div>input {{ border-radius: 12px; padding: 12px; border: 1px solid #ccc; font-size: 16px; }}
        .stButton>button {{ background: {button_color}; color: white; border-radius: 12px; padding: 12px 24px; font-weight: bold; transition: 0.3s; }}
        .stButton>button:hover {{ background: #2563eb; transform: scale(1.05); }}
        .stMarkdown {{ font-size: 18px; color: {text_color}; }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 Chat with Your PDFs - AI Assistant")

    st.markdown("### 🔍 Ask a Question from the PDF Files")
    user_question = st.text_input("", placeholder="Type your question here...", help="Enter your query related to the uploaded PDFs.")

    if user_question:
        with st.spinner("Processing your question..."):
            user_input(user_question)

    with st.sidebar:
        st.header("⚙️ Menu")

        if st.button("🌙 Toggle Dark Mode" if st.session_state.theme == "light" else "☀️ Toggle Light Mode"):
            toggle_theme()
            st.rerun()

        pdf_docs = st.file_uploader("📂 Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process", use_container_width=True):
            with st.spinner("📖 Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete!")

if __name__ == "__main__":
    main()
