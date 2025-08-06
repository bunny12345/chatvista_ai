import os
import io
import json
import tarfile
import tempfile
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from transformers import AutoTokenizer

# Load config from environment variables
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

# Prompt template from environment
PROMPT_TEMPLATE = os.getenv("""PROMPT_TEMPLATE""")

# Basic in-memory cache
CACHE = {}

# Tokenizer for token counting (logging/debugging)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-1B-Instruct")

def count_tokens(text):
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.shape[-1]

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
        model_id=None  # Not needed as vectors are precomputed
    )
    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

def build_prompt(docs, question):
    context_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
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

        # Handle preflight CORS
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

        # Extract the question from body
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

        # Check cache
        if question in CACHE:
            print("Cache hit")
            cached_answer, cached_sources = CACHE[question]
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "answer": cached_answer,
                    "sources": cached_sources,
                    "cached": True
                })
            }

        # Load vectorstore and search
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(question, k=4)
        prompt = build_prompt(docs, question)

        print(f"Token count: {count_tokens(prompt)}")

        # Invoke LLM
        llm_response = call_llm(prompt)

        answer = llm_response.content
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        # Cache result
        CACHE[question] = (answer, sources)

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "answer": answer,
                "sources": sources,
                "cached": False
            })
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
