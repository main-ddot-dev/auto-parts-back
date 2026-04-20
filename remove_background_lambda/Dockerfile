# 1. Python 3.12 on Amazon Linux 2023
FROM public.ecr.aws/lambda/python:3.12

# 2. Copy and install Python dependencies (lightweight — no rembg/numba/scipy)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 3. Bake the u2net model into the image
RUN mkdir -p /var/task/models
RUN python -c "import os; os.environ['U2NET_HOME']='/tmp'; from pooch import retrieve; retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx', known_hash=None, fname='u2net.onnx', path='/tmp')" && \
    mv /tmp/u2net.onnx /var/task/models/u2net.onnx || true
RUN chmod -R 755 /var/task/models

# 4. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
