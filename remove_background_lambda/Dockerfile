# 1. Python 3.12 on Amazon Linux 2023 (modern system libraries and GCC)
FROM public.ecr.aws/lambda/python:3.12

# 2. Install system dependencies for OpenCV and image processing
RUN dnf install -y mesa-libGL gcc-c++ python3-devel && \
    dnf clean all

# 3. Copy and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 4. Environment variables for robustness
ENV NUMBA_CACHE_DIR=/tmp
ENV NUMBA_THREADING_LAYER=workqueue
ENV NUMBA_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# 5. Pre-download AI model in default rembg path and copy to /tmp with full permissions
RUN mkdir -p /home/sbx_user1051/.u2net
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))"
RUN chmod -R 777 /root/.u2net && cp -r /root/.u2net /tmp/ && chmod -R 777 /tmp/.u2net

# 6. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
