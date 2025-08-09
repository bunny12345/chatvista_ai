import boto3
import os
import faiss
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
def load_documents():
    print("🔹 Loading documents from S3...")
    loader = S3FileLoader(bucket="irlcolleges", key="IRL_college_all_data.pdf")  # Replace these
    return loader.load()

def create_embedder():
    print("🔹 Creating Titan embedder...")
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="eu-west-1"
    )

def build_vector_store(docs, embedder):
    print("🔹 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("❌ No chunks created from documents. Check your S3 bucket path and contents.")

    print("🔹 Building FAISS vector store...")
    return FAISS.from_documents(chunks, embedder)

if __name__ == "__main__":
    try:
        docs = load_documents()
        embedder = create_embedder()
        db = build_vector_store(docs, embedder)
        db.save_local("faiss_index")
        print("✅ FAISS vector store saved locally to `faiss_index/`")
    except Exception as e:
        print(f"❌ Error: {e}")