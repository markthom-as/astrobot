from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import pinecone
from langchain.vectorstores import Pinecone

# load environment variables
load_dotenv()

# initialize Pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)

# load files from /pdfs directory and split them into chunks
pdf_reader = PdfReader(
    "pdfs/ancient-astrology-in-theory-and-practice-a-manual--annas-archive (1).pdf"
)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)
chunks = text_splitter.create_documents(text)

# create embeddings
embeddings = OpenAIEmbeddings()

# knowledge_base = FAISS.from_texts(chunks, embeddings)


# Upload vectors to Pinecone
index_name = "astro"
search = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# print search results
print(search.search("Who is Demetra George?"))
