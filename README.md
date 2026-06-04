# ChatVista AI - AWS Lambda Chatbot

A serverless chatbot powered by AWS Lambda, Bedrock LLMs, and FAISS vector search.

## Models Used

- **Embeddings**: Cohere Embed v4 (`cohere.embed-v4:0`)
- **LLM**: Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`)
- **Vector Store**: FAISS (index stored in S3)

## Project Structure

```
chatvista_ai/
├── lambda_function.py          # Main Lambda handler
├── haiku_model.py              # Alternative model with caching
├── main.py                     # FAISS index builder
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container setup
├── terraform/                  # Infrastructure automation
│   ├── provider.tf
│   ├── variables.tf
│   ├── main.tf
│   └── outputs.tf
├── .github/workflows/
│   ├── 00_full_deploy.yml      # Full automation pipeline
│   ├── 01_faiss_index.yml      # Build FAISS index
│   ├── 04_push_to_ecr.yml      # Push to ECR
│   └── 05_lambda_deploy.yml    # Deploy to Lambda
└── README.md
```

## Setup & Deployment

### Local Testing

1. **Build Docker image**:
   ```bash
   docker build -t chatvista-lambda-local .
   ```

2. **Run local test** (with AWS credentials):
   ```bash
   docker run --rm --entrypoint "" \
     -e AWS_ACCESS_KEY_ID=<your-key> \
     -e AWS_SECRET_ACCESS_KEY=<your-secret> \
     -e AWS_DEFAULT_REGION=eu-west-1 \
     chatvista-lambda-local python lambda_function.py
   ```

### GitHub Actions Deployment

You can run the smaller workflows manually, or use the full automation workflow.

- **00_full_deploy.yml** - Full pipeline: build FAISS index, upload to S3, apply Terraform, build/push Docker image, and deploy Lambda.
- **01_faiss_index.yml** - Build FAISS index only.
- **04_push_to_ecr.yml** - Build Docker image and push to ECR.
- **05_lambda_deploy.yml** - Deploy/update Lambda only.

Use the full deployment workflow for the easiest migration and repeatable setup.

## Updating Documents for Vector Search

### Step 1: Update Source Document

Edit the document path in `main.py`:

```python
loader = S3FileLoader(
    bucket="irlcolleges",
    key="YOUR_NEW_DOCUMENT.pdf"  # ← Change this
)
```

### Step 2: Rebuild FAISS Index

The FAISS index must match your embedding model (Cohere):

- Run workflow **01_faiss_index.yml** to generate new index
- Download the `faiss_index.tar.gz` artifact

### Step 3: Upload Index to S3

Replace the existing file in S3:

```bash
aws s3 cp faiss_index.tar.gz s3://faissindexingirlcollege/faiss_index.tar.gz \
  --region eu-west-1
```

Or use AWS Console:
1. Go to S3 bucket `faissindexingirlcollege`
2. Delete old `faiss_index.tar.gz`
3. Upload new `faiss_index.tar.gz`

### Step 4: Deploy Lambda

Run **05_lambda_deploy.yml** to deploy with new index.

## Testing the Chatbot

**Local test event** (in `lambda_function.py`):

```python
test_event = {
    "body": json.dumps({"question": "Your question here?"}),
    "headers": {"Content-Type": "application/json"}
}
```

**Expected response**:

```json
{
  "statusCode": 200,
  "body": {
    "answer": "Response from Claude 3 Sonnet",
    "sources": ["s3://bucket/document.pdf"]
  }
}
```

## AWS Configuration

Required:
- S3 bucket: `faissindexingirlcollege` (FAISS index storage)
- S3 bucket: `irlcolleges` (Source documents)
- ECR repository for Docker images
- Lambda execution IAM role with:
  - S3 read access
  - Bedrock InvokeModel access
  - CloudWatch Logs write access

If you use the full Terraform automation, these resources are created by the `terraform/` configuration.

## Troubleshooting

**"Missing expected file: index.faiss"**
- FAISS tar.gz corrupted or incorrectly formatted
- Rebuild using: `tar -czvf faiss_index.tar.gz faiss_index`

**"Embedding dimension mismatch"**
- FAISS index built with different embedding model
- Rebuild index with `cohere.embed-v4:0`

**"Model is Legacy"**
- Claude 3 Haiku no longer active
- Using Claude 3 Sonnet (current stable version)

## Environment Variables

- `AWS_REGION`: `eu-west-1`
- `S3_BUCKET`: `faissindexingirlcollege`
- `S3_KEY`: `faiss_index.tar.gz`
- `EMBED_MODEL_ID`: `cohere.embed-v4:0`
- `LLM_MODEL_ID`: `anthropic.claude-3-sonnet-20240229-v1:0`

## GitHub Secrets for Full Deployment

The full workflow uses these secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `FAISS_S3_BUCKET`
