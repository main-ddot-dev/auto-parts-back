import os
import shutil
import boto3
import io
import logging
from PIL import Image

# --- CRITICAL: environment setup BEFORE any rembg/numba import ---
# Copy pre-compiled Numba cache from read-only /var/task to writable /tmp
tmp_cache = '/tmp/numba_cache'
precompiled_cache = '/var/task/precompiled_numba'

if not os.path.exists(tmp_cache):
    os.makedirs(tmp_cache, exist_ok=True)
    if os.path.exists(precompiled_cache):
        for item in os.listdir(precompiled_cache):
            src = os.path.join(precompiled_cache, item)
            dst = os.path.join(tmp_cache, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

os.environ['NUMBA_CACHE_DIR'] = tmp_cache
os.environ['U2NET_HOME'] = '/tmp'
os.environ['OMP_NUM_THREADS'] = '1'

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global session to persist the AI model across invocations (warm starts)
session = None

def handler(event, context):
    global session
    from rembg import remove, new_session

    # 1. Load pre-installed model with pre-compiled cache
    model_path = "/var/task/models/u2net.onnx"

    if session is None:
        logger.info("Initializing AI session with pre-compiled cache...")
        session = new_session("u2net", model_path=model_path)

    # 2. S3 event data
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # 3. Download image from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        input_image = Image.open(io.BytesIO(response['Body'].read()))

        # 4. AI background removal
        logger.info(f"Processing: {key}")
        output_image = remove(input_image, session=session)

        # 5. Save as PNG and upload
        out_buffer = io.BytesIO()
        output_image.save(out_buffer, format='PNG')
        out_buffer.seek(0)

        dest_bucket = os.environ['DESTINATION_BUCKET']
        dest_prefix = os.environ['DESTINATION_PREFIX']
        new_key = f"{dest_prefix}{os.path.basename(key).rsplit('.', 1)[0]}.png"

        s3.put_object(Bucket=dest_bucket, Key=new_key, Body=out_buffer, ContentType='image/png')
        logger.info(f"Success: image saved to {new_key}")

        return {
            "statusCode": 200,
            "body": f"Success: {key} processed and saved to {new_key}"
        }

    except Exception as e:
        logger.error(f"Error processing the image: {str(e)}", exc_info=True)
        raise e
