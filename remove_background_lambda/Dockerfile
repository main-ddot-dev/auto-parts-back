# 1. Python 3.12 on Amazon Linux 2023 (modern system libraries and GCC)
FROM public.ecr.aws/lambda/python:3.12

# 2. Install system dependencies for OpenCV and image processing
RUN dnf install -y mesa-libGL gcc-c++ python3-devel && dnf clean all

# 3. Copy and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 4. Total Numba exclusion — prevent any filesystem access attempts
ENV NUMBA_DISABLE_CACHE=1
ENV NUMBA_THREADING_LAYER=workqueue
ENV OMP_NUM_THREADS=1
ENV HOME=/tmp
ENV NUMBA_CACHE_DIR=/tmp

# 5. Bake AI model and clean up any Numba cache leftovers from the build
RUN mkdir -p /var/task/models
RUN python -c "import os; os.environ['U2NET_HOME']='/tmp'; from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))" && \
    mv /tmp/u2net.onnx /var/task/models/u2net.onnx && \
    rm -rf /tmp/numba_cache /root/.cache/numba
RUN chmod -R 755 /var/task/models

# 6. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
