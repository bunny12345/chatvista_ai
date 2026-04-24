FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY haiku_model.py .
COPY requirements.txt .

# Install build dependencies required for compiling scipy, numpy, and other packages
RUN yum install -y centos-release-scl && \
    yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-make lapack-devel blas-devel && \
    source /opt/rh/devtoolset-9/enable && \
    echo "source /opt/rh/devtoolset-9/enable" >> ~/.bashrc

# Install Python packages
RUN source /opt/rh/devtoolset-9/enable && \
    pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Set the Lambda handler (filename.function_name)
CMD ["haiku_model.lambda_handler"]