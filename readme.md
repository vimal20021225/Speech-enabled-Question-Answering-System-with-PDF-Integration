# Speech-enabled Question Answering System with PDF Integration

## Introduction
------------
The Speech-enabled Question Answering System with PDF Integration is a Python application that combines the capabilities of natural language processing and speech recognition to allow users to ask questions about multiple PDF documents using their voice. This system extracts text content from PDFs and generates accurate responses to user queries based on the content of the documents. Please note that the system will only respond to questions related to the loaded PDFs.

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


## Contributing
------------
This repository is intended for educational purposes and does not accept further contributions. It serves as supporting material for a YouTube tutorial that demonstrates how to build this project. Feel free to utilize and enhance the app based on your own requirements.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).
