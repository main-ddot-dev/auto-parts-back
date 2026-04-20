# 1. Python 3.12 on Amazon Linux 2023 (modern system libraries and GCC)
FROM public.ecr.aws/lambda/python:3.12

# 2. Install system dependencies for OpenCV and image processing
RUN dnf install -y mesa-libGL gcc-c++ python3-devel && dnf clean all

# 3. Copy and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 4. Fixed Numba cache directory (pre-compiled during build)
ENV NUMBA_CACHE_DIR=/var/task/numba_cache
RUN mkdir -p $NUMBA_CACHE_DIR

# 5. Download AI model, pre-compile Numba, and move model to fixed path
RUN mkdir -p /var/task/models
RUN python -c "import os; os.environ['U2NET_HOME']='/tmp'; from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))" && \
    mv /tmp/u2net.onnx /var/task/models/u2net.onnx

# 6. Full permissions so Lambda user can read Numba cache and model
RUN chmod -R 777 $NUMBA_CACHE_DIR /var/task/models

# 7. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
