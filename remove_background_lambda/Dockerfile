# 1. Python 3.12 on Amazon Linux 2023 (modern system libraries and GCC)
FROM public.ecr.aws/lambda/python:3.12

# 2. Install system dependencies for OpenCV and image processing
# On AL2023 we use 'dnf' instead of 'yum'
RUN dnf install -y mesa-libGL gcc-c++ python3-devel && \
    dnf clean all

# 3. Copy dependencies and upgrade pip
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --upgrade pip

# 4. Install Python libraries
RUN pip install -r requirements.txt

# 5. Environment variables for robustness
ENV NUMBA_CACHE_DIR=/tmp
ENV NUMBA_THREADING_LAYER=workqueue
ENV U2NET_HOME=/var/task/.u2net
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))"

# 6. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
