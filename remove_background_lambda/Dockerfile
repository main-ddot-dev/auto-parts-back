# 1. Use the official AWS image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# 2. Copy dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Install everything (rembg, Pillow, boto3)
RUN pip install -r requirements.txt

# 4. Set the model path and pre-download it along with the code
ENV U2NET_HOME=${LAMBDA_TASK_ROOT}/.u2net
RUN python -c "from rembg import remove; from PIL import Image; remove(Image.new('RGB', (1,1)))"

# 5. Copy our code
COPY app.py ${LAMBDA_TASK_ROOT}

# 6. Tell AWS which function to start
CMD [ "app.handler" ]
