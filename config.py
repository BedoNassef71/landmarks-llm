import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Use environment variables for sensitive data
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

new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)