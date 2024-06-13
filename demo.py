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
from htmlTemplates import css, bot_template, user_template
import pyttsx3
import speech_recognition
import pygame
from gtts import gTTS
import json

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pygame.mixer.init()

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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write(bot_template.replace("{{MSG}}", response["output_text"]), unsafe_allow_html=True)
    return response["output_text"]

def voiceinput():
    r = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source, 0, 4)
    try:
        query = r.recognize_google(audio, language='en-in')
    except:
        return ""
    return query

def play_audio(file_path):
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def main():
    st.set_page_config("VOICE ED")
    st.header("VOICE ED")

    # Load existing questions and answers
    if os.path.exists("qa_store.json"):
        with open("qa_store.json", "r") as f:
            qa_store = json.load(f)
    else:
        qa_store = {}

    user_question = st.chat_input("Ask your Question")
    start = st.button("START")
    if user_question:
        text = user_input(user_question)
        filename = f"{user_question[:30].replace(' ', '_')}.mp3"
        t_t_s = gTTS(text=text, lang='en', slow=False)
        t_t_s.save(filename)
        qa_store[filename] = {"question": user_question, "answer": text}
        with open("qa_store.json", "w") as f:
            json.dump(qa_store, f)
        play_audio(filename)

    if start:
        while True:
            query = voiceinput().lower()
            if "wake up" in query:
                t_t_s = gTTS(text="Hello, how can I help you?", lang='en', slow=False)
                t_t_s.save('wake_up.mp3')
                play_audio('wake_up.mp3')
                while True:
                    user_query = voiceinput().lower()
                    if "tell me" in user_query:
                        user_query = user_query.replace("tell me", "")
                        text = user_input(user_query)
                        filename = f"{user_query[:30].replace(' ', '_')}.mp3"
                        t_t_s = gTTS(text=text, lang='en', slow=False)
                        t_t_s.save(filename)
                        qa_store[filename] = {"question": user_query, "answer": text}
                        with open("qa_store.json", "w") as f:
                            json.dump(qa_store, f)
                        play_audio(filename)
                    if "go to sleep" in user_query:
                        t_t_s = gTTS(text="Ok, sir. You can call me any time.", lang='en', slow=False)
                        t_t_s.save('sleep.mp3')
                        play_audio('sleep.mp3')
                        break

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.sidebar.title("Audio Files")
        audio_files = [f for f in os.listdir() if f.endswith('.mp3')]
        selected_audio = st.sidebar.selectbox("Select an audio file to play", audio_files)
        if st.sidebar.button("Play Selected Audio"):
            if selected_audio in qa_store:
                st.write("**Question:**", qa_store[selected_audio]["question"])
                st.write("**Answer:**", qa_store[selected_audio]["answer"])
            play_audio(selected_audio)

if __name__ == "__main__":
    main()