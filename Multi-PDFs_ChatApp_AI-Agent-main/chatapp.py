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
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Check available models and select a supported one
models = genai.list_models()
available_models = [model.name for model in models if "generateContent" in model.supported_generation_methods]
if not available_models:
    st.error("No supported models found. Please check your API key or Google Generative AI availability.")
    st.stop()
# Prioritize gemini-1.5-pro, fallback to gemini-1.5-flash, or use the first available
selected_model = next((model for model in ["models/gemini-1.5-pro", "models/gemini-1.5-flash"] if model in available_models), available_models[0])
print(f"Using model: {selected_model}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context," don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)    
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    time.sleep(2)  # Stay within rate limits
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    answer = response["output_text"]

    # Store question and answer in session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"question": user_question, "answer": answer})

    return answer

def display_chat_history():
    st.subheader("📜 Chat History")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):  # Show recent chats first
            st.markdown(
                f"""
                <div style='padding: 10px; margin: 10px 0; background: #f4f4f4; border-radius: 10px;'>
                    <strong>Q: {chat['question']}</strong>
                    <br>
                    <span style='color: #333;'>💬 {chat['answer']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No chat history available. Start asking questions!")

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    # Sidebar for PDF upload
    with st.sidebar:
        st.image("img/bot.jpg")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.write("---")

    # Main chat area
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    
    if user_question:
        answer = user_input(user_question)
        st.write("Reply: ", answer)

    display_chat_history()

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #B2BEB5; padding: 15px; text-align: center;">
            PDF CHATBOT ❤️ [ DIV - KAV - HAR - IND]
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
