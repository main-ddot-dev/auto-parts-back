# 1. Python 3.12 on Amazon Linux 2023
FROM public.ecr.aws/lambda/python:3.12

# 2. Install wget and graphics libraries
RUN dnf install -y mesa-libGL wget && dnf clean all

# 3. Set working directory to Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Download u2net model directly via wget
RUN mkdir -p models && \
    wget -q https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O models/u2net.onnx
RUN chmod -R 755 models

# 6. Copy code and set entrypoint
COPY app.py .
CMD [ "app.handler" ]
