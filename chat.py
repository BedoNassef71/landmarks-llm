from flask import Flask, request, jsonify
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

os.environ['GOOGLE_API_KEY'] = "AIzaSyCTMXGCg2BFuyqpCYpJ2E1eg-rST9p3GWk"

# Initialize models and objects outside the function to reuse them
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
prompt_template = """
Answer the question as full detailed as possible from the provided context \n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_conversational_chain():
    return chain


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def get_class_info():
    data = request.get_json()
    class_name = data['class_name']
    result = user_input(class_name)
    return jsonify(result)


def user_input(class_name):
    loader = DirectoryLoader('texts', glob="*" + class_name + "*/*.txt")
    raw_text = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=2816, chunk_overlap=256)
    documents = text_splitter.split_documents(raw_text)

    # Use the pre-initialized embeddings object
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    user_question = "Give me info about " + class_name

    docs = vector_store.similarity_search(user_question)

    # Use the pre-initialized conversational chain
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response


if __name__ == '__main__':
    app.run(debug=True)
