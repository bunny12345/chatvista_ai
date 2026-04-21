FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY haiku_model.py .
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Set the Lambda handler (filename.function_name)
CMD ["haiku_model.lambda_handler"]