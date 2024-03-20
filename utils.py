from config import chain, embeddings, new_db
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

def ans_question(user_question):
    docs = new_db.similarity_search(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    return response

def user_input(class_name):
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import CharacterTextSplitter
    from config import embeddings

    loader = DirectoryLoader('texts', glob="*" + class_name + "*/*.txt")
    raw_text = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2816, chunk_overlap=256)
    documents = text_splitter.split_documents(raw_text)
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    user_question = "Give me info about " + class_name
    docs = vector_store.similarity_search(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    return response
