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
ENV NUMBA_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV U2NET_HOME=/tmp/.u2net

# 6. Download AI model during build, then fix permissions
RUN mkdir -p /var/task/.u2net
RUN python -c "import os; os.environ['U2NET_HOME']='/var/task/.u2net'; from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))"
RUN chmod -R 777 /var/task/.u2net

# 7. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
