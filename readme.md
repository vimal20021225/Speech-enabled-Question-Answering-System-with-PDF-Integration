# Speech-enabled Question Answering System with PDF Integration

# Voice ED

## Project Overview
Voice ED is a web application that leverages AI to provide voice-based question answering from custom data contained in PDF documents. It is designed for educators, researchers, and anyone needing efficient, voice-activated information retrieval from their documents. Voice ED presents a convenient solution for extracting and answering questions based on your text materials.

### Features
- **Voice Interaction:** Allows users to interact with the application using voice commands to ask questions and receive spoken answers.
- **Document Upload:** Supports uploading PDF files for processing and information extraction.
- **Question Answering:** Provides detailed answers to user queries based on the content of uploaded PDFs.
- **Audio Responses:** Delivers answers in audio format using text-to-speech conversion.
- **Wake Word Activation:** Allows voice-activated interaction with a predefined wake word.

## Technologies Used
- **Programming Languages:** Python
- **Libraries and Frameworks:** Streamlit, FAISS, langchain, langchain_google_genai, dotenv, PyPDF2, gTTS, pygame, speech_recognition
- **Cloud Platform:** Google Cloud Platform (for Google Generative AI API)

## NLP Algorithm
Utilizes pre-trained language models through the ChatGoogleGenerativeAI model from langchain_google_genai to generate detailed answers to user queries.

## Document Parsing
- **PDF Parsing:** Uses PyPDF2 to extract text from PDF files.

## Voice Interaction
- **Speech Recognition:** Utilizes the speech_recognition library to convert spoken queries into text.
- **Text-to-Speech:** Uses gTTS and pygame to convert text responses into spoken audio.

## Streamlit Interface
Uses Streamlit to create an interactive web interface, allowing users to upload PDF documents, ask questions via text or voice, and listen to audio responses.

## Error Handling
Includes robust error handling to detect, log, and display informative error messages.

## Question Answering Process
Parses uploaded documents, extracts text, and generates detailed answers to user queries based on the context of the documents.

## Deployment
The application is deployed on a cloud platform, making it easily accessible to users.
## How It Works
------------

PDF Loading: The system reads multiple PDF documents and extracts their text content.

Text Chunking: The extracted text is segmented into smaller chunks for effective processing.

Speech Recognition: Users can ask questions verbally using the system's speech recognition feature.

Natural Language Processing: The system utilizes a language model to process the user's question and extract its semantic meaning.

Similarity Matching: The system compares the user's question with the text chunks from the PDFs to identify the most relevant content.

Response Generation: Based on the identified relevant content, the system generates a response to the user's question.

## Dependencies and Installation
----------------------------
please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from Hugging Face and add it to the `.env` file in the project directory.
```commandline
HUGGING FACE=your_secrit_api_key
```
