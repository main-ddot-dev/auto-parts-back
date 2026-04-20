# 1. Python 3.12 on Amazon Linux 2023 (modern system libraries and GCC)
FROM public.ecr.aws/lambda/python:3.12

# 2. Install system dependencies for OpenCV and image processing
RUN dnf install -y mesa-libGL gcc-c++ python3-devel && dnf clean all

# 3. Copy and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 4. Build-time environment variables
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV OMP_NUM_THREADS=1

# 5. Download model (defaults to /root/.u2net/), generate Numba cache, move to fixed paths
RUN mkdir -p /var/task/models /var/task/precompiled_numba /tmp/numba_cache
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))" && \
    mv /root/.u2net/u2net.onnx /var/task/models/u2net.onnx && \
    cp -r /tmp/numba_cache/* /var/task/precompiled_numba/
RUN chmod -R 755 /var/task/models /var/task/precompiled_numba

# 6. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
