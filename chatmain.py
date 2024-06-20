import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import time
from fpdf import FPDF
from instructor import Instructor
 
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API key not found. Please set the 'GOOGLE_API_KEY' environment variable.")
    st.stop()
 
genai.configure(api_key=google_api_key)
 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain
 
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
   
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
 
    chain = get_conversational_chain()
 
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]
 
# Function to recognize speech using Google Speech Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
 
    try:
        st.sidebar.write("Recognizing...")
        text = recognizer.recognize_google(audio_data, key=google_api_key)
        st.sidebar.write("You said:", text)
        return text
    except sr.UnknownValueError:
        st.sidebar.write("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        st.sidebar.write(f"Could not request results from Google Speech Recognition service; {e}")
 
# Function to create and save a PDF of the search history
def save_search_history_to_pdf(search_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
 
    pdf.cell(200, 10, txt="Search History", ln=True, align='C')
 
    for entry in search_history:
        pdf.cell(200, 10, txt=f"Question: {entry['question']}", ln=True, align='L')
        pdf.multi_cell(0, 10, txt=f"Response: {entry['response']}")
        pdf.ln(5)
 
    pdf_file = "search_history.pdf"
    pdf.output(pdf_file)
    return pdf_file
 
# Main function
def main():
    st.set_page_config(page_title="NoblQ Chatbot", page_icon=":robot:", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    body {
        color: black;
        background-color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    st.sidebar.title("Menu")
   
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
   
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
 
    if "question_asked" not in st.session_state:
        st.session_state.question_asked = False
   
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
 
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
 
    if pdf_docs and not st.session_state.file_uploaded:
        upload_warning = st.sidebar.empty()
        upload_warning.warning("Files are uploaded", icon="✅")
        time.sleep(3)
        upload_warning.empty()
        st.session_state.file_uploaded = True
 
    selected_pdf = st.sidebar.selectbox("Select a PDF to view", pdf_docs, format_func=lambda x: x.name if x else "None")
 
    if pdf_docs and st.sidebar.button("Process PDFs"):
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.sidebar.success("Processing Complete")
            process_warning = st.sidebar.empty()
            process_warning.warning("PDFs are processed", icon='✅')
            time.sleep(3)
            process_warning.empty()
            st.session_state.pdf_processed = True
 
    st.title("Ask a question:")
 
    col1, col2 = st.columns(2)
 
    with col1:
        user_question = st.text_input("Your Question")
 
        if user_question:
            with st.spinner("Processing..."):
                response = user_input(user_question)
                st.write("Reply:", response)
                if "This question cannot be answered from the given context" in response:
                    st.warning("Failed to retrieve. Try again!")
                else:
                    st.warning("Retrieved successfully!")
                    st.session_state.search_history.append({"question": user_question, "response": response})
 
        st.sidebar.markdown("---")
        st.sidebar.title("Voice Assistant")
        st.sidebar.write("Click the button below and speak to ask a question:")
        if st.sidebar.button("Start Voice Assistant"):
            command = recognize_speech()
            if command:
                if "stop" in command.lower():
                    st.sidebar.write("Voice Assistant Stopped.")
                else:
                    st.sidebar.write("You said:", command)
                    with st.spinner("Processing..."):
                        response = user_input(command)
                        st.write("Reply:", response)
                        if "This question cannot be answered from the given context" in response:
                            st.warning("Failed to retrieve. Try again!")
                        else:
                            st.warning("Retrieved successfully!")
                            st.session_state.search_history.append({"question": command, "response": response})
 
    st.markdown("---")  
 
    if not user_question:  
        with col2:
            if selected_pdf:
                st.write(f"Displaying contents of {selected_pdf.name}:")
                pdf_reader = PdfReader(selected_pdf)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                st.text_area("PDF Content", pdf_text, height=300)
 
    if "error" in st.session_state:
        st.sidebar.error(st.session_state.error)
 
    st.sidebar.markdown("## Search History")
    with st.sidebar.expander("Click to view search history"):
        for entry in st.session_state.search_history:
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Response:** {entry['response']}")
            st.markdown("---")
   
    if st.sidebar.button("Download Search History as PDF"):
        pdf_file = save_search_history_to_pdf(st.session_state.search_history)
        with open(pdf_file, "rb") as f:
            st.sidebar.download_button(
                label="Download PDF",
                data=f,
                file_name="search_history.pdf",
                mime="application/pdf"
            )
 
if __name__ == "__main__":
    main()