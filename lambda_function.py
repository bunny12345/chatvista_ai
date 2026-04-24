import os
import io
import json
import tarfile
import tempfile
import traceback
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock

# Configs
S3_BUCKET = "faissindexingirlcollege"
S3_KEY = "faiss_index.tar.gz"
AWS_REGION = "eu-west-1"

# Bedrock model IDs
EMBED_MODEL_ID = "cohere.embed-v4:0"
LLM_MODEL_ID = "arn:aws:bedrock:eu-west-1:931886962745:inference-profile/eu.anthropic.claude-3-haiku-20240307-v1:0"

# Simple in-memory cache for question -> answer
CACHE = {}

def download_and_extract_faiss():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)

    temp_dir = tempfile.mkdtemp()
    tar_data = io.BytesIO(response["Body"].read())

    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
        members = tar.getmembers()
        in_subdir = any(m.name.startswith("faiss_index/") for m in members if m.isfile())

        for member in members:
            if member.isfile():
                if in_subdir and member.name.startswith("faiss_index/"):
                    member.name = member.name[len("faiss_index/"):]
                tar.extract(member, path=temp_dir)

    for file_name in ["index.faiss", "index.pkl"]:
        if not os.path.exists(os.path.join(temp_dir, file_name)):
            raise FileNotFoundError(f"Missing expected file: {file_name}")

    return temp_dir

def load_vectorstore():
    temp_dir = download_and_extract_faiss()
    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        model_id=EMBED_MODEL_ID,
        provider="cohere"
    )
    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

def build_prompt(docs, question):
    template = """You are a concise and helpful assistant.
- Answer briefly and clearly using no more than 2 short paragraphs.
- Avoid repetition or over-explaining.
- Limit your response to 300 tokens maximum."
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=question)

def call_llm(prompt):
    model = ChatBedrock(
        model_id=LLM_MODEL_ID,
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        provider="anthropic"
    )
    return model.invoke(prompt)

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({"message": "CORS preflight success"})
            }

        question = event.get("question")
        if not question and "body" in event:
            body = json.loads(event["body"])
            question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'question'"})
            }

        # Check cache first
        if question in CACHE:
            print("Returning cached answer")
            cached_response = CACHE[question]
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps(cached_response)
            }

        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(question, k=4)
        print("Docs loaded:", len(docs))
        print("Doc sources:", [doc.metadata.get("source", "unknown") for doc in docs])

        prompt = build_prompt(docs, question)
        print("Prompt length:", len(prompt))
        llm_response = call_llm(prompt)
        print("LLM response type:", type(llm_response))
        print("LLM response repr:", repr(llm_response)[:1000])

        response_body = {
            "answer": getattr(llm_response, 'content', None),
            "sources": list({doc.metadata.get("source", "unknown") for doc in docs})
        }

        # Cache the answer
        CACHE[question] = response_body

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps(response_body)
        }

    except Exception as e:
        error_message = str(e) or repr(e)
        print("Error type:", type(e).__name__)
        print("Error message:", error_message)
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": error_message})
        }
