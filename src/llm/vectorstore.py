import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ['GOOGLE_API_KEY'] = "AIzaSyCTMXGCg2BFuyqpCYpJ2E1eg-rST9p3GWk"

loader = DirectoryLoader('../../texts', glob="**/*.txt")
raw_text = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
documents = text_splitter.split_documents(raw_text)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(documents, embedding=embeddings)
vector_store.save_local("faiss_index")
