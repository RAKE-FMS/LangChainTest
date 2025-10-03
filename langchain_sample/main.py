import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Ensure your Google API key is set as an environment variable
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and set if not using .env

import google.generativeai as genai
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

# Load the CSV file
loader = CSVLoader(file_path="langchain_sample/sample.csv")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a Chroma vector store
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

# Create a retriever
retriever = vectordb.as_retriever()

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Query the CSV data
query = "What is Alice's age?"
response = qa_chain.invoke({"query": query})
print(response["result"])

query = "Which city does Bob live in?"
response = qa_chain.invoke({"query": query})
print(response["result"])
