# 1. Official AWS image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# 2. Install compiler and required system dependencies
# gcc and python3-devel for compiling; mesa-libGL for OpenCV (rembg)
RUN yum install -y gcc gcc-c++ python3-devel mesa-libGL && \
    yum clean all

# 3. Copy dependencies and upgrade pip
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --upgrade pip

# 4. Install Python libraries
RUN pip install -r requirements.txt

# 5. Pre-download the AI model
ENV U2NET_HOME=${LAMBDA_TASK_ROOT}/.u2net
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))"

# 6. Copy code and set entrypoint
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]
