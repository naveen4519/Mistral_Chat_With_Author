from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from bson import ObjectId
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import logging

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Load environment variables
load_dotenv()

# MongoDB setup
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise ValueError("MONGODB_URI not found in environment variables")
client = MongoClient(mongodb_uri)
db = client["pdf_chat"]
chat_sessions_collection = db["chat_sessions"]

# Global variables to store processed data
vector_store = None
chain = None

translator = Translator()

def calculate_cosine_similarity(text, user_question):
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform([text, user_question])
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cos_similarity

def generate_pdf_title(text):
    words = text.split()
    return ' '.join(words[:5]) + "..."

@app.route('/new_chat', methods=['POST'])
def new_chat():
    global vector_store, chain
    logger.info('Received request for new chat')
    if 'file' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f'File saved: {filepath}')

            # Process the PDF
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            logger.info('PDF loaded')

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            text_chunks = text_splitter.split_documents(documents)
            logger.info('Text split into chunks')

            embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
            vector_store = FAISS.from_documents(text_chunks, embeddings)
            logger.info('Vector store created')

            llm = CTransformers(model=r"C:\Users\nakir\Downloads\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                    config={'max_new_tokens': 500, 'temperature': 0.1})
            
            logger.info('Language model loaded')

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                          retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                          memory=memory)
            logger.info('Conversational chain created')

            # Generate a title for the PDF
            pdf_title = generate_pdf_title(documents[0].page_content)

            # Create a new chat session in MongoDB
            chat_session = {
                "pdf_title": pdf_title,
                "questions_answers": [],
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            result = chat_sessions_collection.insert_one(chat_session)
            chat_id = str(result.inserted_id)
            logger.info(f'New chat session created with ID: {chat_id}')

            return jsonify({'message': 'New chat created successfully', 'chat_id': chat_id, 'pdf_title': pdf_title}), 200
        except Exception as e:
            logger.error(f'Error processing PDF: {str(e)}')
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    else:
        logger.error('Invalid file type')
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400






@app.route('/ask', methods=['POST'])
def ask_question():
    global chain, vector_store
    logger.info('Received question request')
    if not chain or not vector_store:
        logger.error('No PDF processed yet')
        return jsonify({'error': 'No PDF processed yet'}), 400

    data = request.json
    question = data.get('question')
    language_code = data.get('language_code', 'en')
    chat_id = data.get('chat_id')

    if not question or not chat_id:
        logger.error('No question or chat_id provided')
        return jsonify({'error': 'No question or chat_id provided'}), 400

    try:
        logger.info(f'Processing question: {question}')
        docs = vector_store.similarity_search(question, k=1)

        if not docs or docs[0].page_content.strip() == "":
            logger.info('Question not found in PDF')
            answer = generate_general_answer(question)
            answer = "Note: This answer is not based on the PDF content. " + answer
        else:
            similarity = calculate_cosine_similarity(docs[0].page_content, question)
            logger.info(f'Similarity score: {similarity}')

            if similarity <= 0.0125:
                logger.info('Question not directly related to PDF content')
                answer = generate_general_answer(question)
                answer = "Note: This answer may not be directly related to the PDF content. " + answer
            else:
                result = chain({"question": question})
                answer = result["answer"]
                logger.info('Answer generated from PDF content')

        translated_answer = translator.translate(answer, dest=language_code).text
        logger.info(f'Answer translated to {language_code}')

        #adding queation,answer pair to mongodb
        chat_sessions_collection.update_one(
            {"_id": ObjectId(chat_id)},
            {
                "$push": {"questions_answers": {"question": question, "answer": translated_answer}},
                "$set": {"updated_at": datetime.now()}
            }
        )
        logger.info('Q&A pair added to chat session')

        return jsonify({'answer': translated_answer}), 200
    except Exception as e:
        logger.error(f'Error processing question: {str(e)}')
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

def generate_general_answer(question):
    # Use a more direct prompt to get a helpful answer
    prompt = f"Please provide a helpful and informative answer to the following question, based on your general knowledge: {question}"
    result = chain({"question": prompt})
    return result["answer"]




@app.route('/chat_sessions', methods=['GET'])
def get_chat_sessions():
    logger.info('Fetching chat sessions')
    try:
        sessions = list(chat_sessions_collection.find())
        for session in sessions:
            session['_id'] = str(session['_id'])  # Convert ObjectId to string
        logger.info(f'Retrieved {len(sessions)} chat sessions')
        return jsonify(sessions)
    except Exception as e:
        logger.error(f'Error fetching chat sessions: {str(e)}')
        return jsonify({"error": "Failed to fetch chat sessions"}), 500

@app.route('/chat_history/<chat_id>', methods=['GET'])
def get_chat_history(chat_id):
    logger.info(f'Fetching chat history for chat ID: {chat_id}')
    try:
        chat_session = chat_sessions_collection.find_one({"_id": ObjectId(chat_id)})
        if chat_session:
            chat_session['_id'] = str(chat_session['_id'])
            logger.info('Chat history retrieved successfully')
            return jsonify(chat_session)
        else:
            logger.warning(f'Chat session not found for ID: {chat_id}')
            return jsonify({"error": "Chat session not found"}), 404
    except Exception as e:
        logger.error(f'Error retrieving chat history: {str(e)}')
        return jsonify({"error": "Failed to retrieve chat history"}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
