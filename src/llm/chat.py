import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

os.environ['GOOGLE_API_KEY'] = "AIzaSyCTMXGCg2BFuyqpCYpJ2E1eg-rST9p3GWk"

def get_conversational_chain():
    prompt_template = """
    Answer the question as full detailed as possible from the provided context \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.9)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(class_name):
    loader = DirectoryLoader('../../texts', glob="*" + class_name + "*/*.txt")
    raw_text = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    documents = text_splitter.split_documents(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    user_question = "Give me info about " + class_name

    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response

ans = user_input("Ahmose")
print(ans)