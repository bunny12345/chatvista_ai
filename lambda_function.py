import os
import io
import json
import tarfile
import re
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate

# Configs
S3_BUCKET = "faissindexingirlcollege"
S3_KEY = "faiss_index.tar.gz"
AWS_REGION = "eu-west-1"

# Bedrock model IDs
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_INFERENCE_PROFILE_ARN = "arn:aws:bedrock:eu-west-1:931886962745:inference-profile/eu.meta.llama3-2-3b-instruct-v1:0"

# Simple in-memory cache dictionary
CACHE = {}
VECTORSTORE = None  # Global cache for FAISS index

# Common CORS headers
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "OPTIONS,POST"
}

def download_and_extract_faiss():
    """Download FAISS index from S3 and extract to /tmp for reuse."""
    local_dir = "/tmp/faiss_index"
    if os.path.exists(os.path.join(local_dir, "index.faiss")) and os.path.exists(os.path.join(local_dir, "index.pkl")):
        print("Using cached FAISS index from /tmp")
        return local_dir

    print("Downloading FAISS index from S3...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)

    os.makedirs(local_dir, exist_ok=True)
    tar_data = io.BytesIO(response["Body"].read())

    with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
        members = tar.getmembers()
        in_subdir = any(m.name.startswith("faiss_index/") for m in members if m.isfile())
        for member in members:
            if member.isfile():
                if in_subdir and member.name.startswith("faiss_index/"):
                    member.name = member.name[len("faiss_index/"):]
                tar.extract(member, path=local_dir)

    for file_name in ["index.faiss", "index.pkl"]:
        if not os.path.exists(os.path.join(local_dir, file_name)):
            raise FileNotFoundError(f"Missing expected file: {file_name}")

    return local_dir

def load_vectorstore():
    """Load FAISS index into memory (cached globally for Lambda warm starts)."""
    global VECTORSTORE
    if VECTORSTORE:
        print("Using in-memory cached FAISS vectorstore")
        return VECTORSTORE

    embeddings = BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        model_id=EMBED_MODEL_ID
    )
    VECTORSTORE = FAISS.load_local(
        download_and_extract_faiss(),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return VECTORSTORE

def build_prompt(docs, question):
    """Build a short, token-limited prompt."""
    template = """You are a concise and helpful assistant.
- Limit your response to 300 tokens maximum.
- Avoid repetition, unnecessary politeness, or extra remarks.
- Only answer using the provided context.

Context:
{context}

Question: {question}

Answer:"""
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context_text, question=question)

def clean_response(text):
    """Remove filler and closing phrases."""
    patterns_to_remove = [
        r"(Let me know.*?)(\n|$)",
        r"(Please let me know.*?)(\n|$)",
        r"(I am here to help.*?)(\n|$)",
        r"(Best regards,.*?)(\n|$)",
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def call_llm(prompt):
    """Call Bedrock LLM with token limit."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    payload = {
        "prompt": prompt,
        "max_gen_len": 300  # Correct param for Llama models in Bedrock
    }

    response = client.invoke_model(
        modelId=LLM_INFERENCE_PROFILE_ARN,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8")
    )
    output = response["body"].read().decode()

    try:
        parsed = json.loads(output)
        raw_text = parsed.get("generation", output)import os
import io
import json
import tarfile
import tempfile
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
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

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
        model_id=EMBED_MODEL_ID
    )
    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

def build_prompt(docs, question):
    template = """You are a concise and helpful assistant.
- Answer briefly and clearly using no more than 2 short paragraphs.
- Avoid repetition or over-explaining.
- If the user says greetings like "hi", "hello", "hey", simply respond with "Hello! How can I help you today?"
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
        client=boto3.client("bedrock-runtime", region_name=AWS_REGION)
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
        prompt = build_prompt(docs, question)
        llm_response = call_llm(prompt)

        response_body = {
            "answer": llm_response.content,
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
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
    except json.JSONDecodeError:
        raw_text = output

    return clean_response(raw_text)

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"message": "CORS preflight success"})}

        # Extract question
        question = event.get("question")
        if not question and "body" in event:
            body = json.loads(event["body"])
            question = body.get("question")

        if not question:
            return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": "Missing 'question'"})}

        # Return cached response if exists
        if question in CACHE:
            print("Returning cached answer")
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(CACHE[question])}

        # Load vectorstore and search
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(question, k=2)  # Reduced for speed
        prompt = build_prompt(docs, question)

        # Call LLM
        llm_response = call_llm(prompt)

        response_body = {
            "answer": llm_response,
            "sources": list({doc.metadata.get("source", "unknown") for doc in docs}),
        }

        # Cache response
        CACHE[question] = response_body

        return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(response_body)}

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    # For local testing
    test_event = {
        "question": "do we get any scholarships in Trinity college?"
    }
    print(lambda_handler(test_event, None))