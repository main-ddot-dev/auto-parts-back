import os
import boto3
import io
import logging
from PIL import Image

# Environment config BEFORE any other imports
os.environ['NUMBA_CACHE_DIR'] = '/var/task/numba_cache'
os.environ['U2NET_HOME'] = '/tmp'

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global session to persist the AI model across invocations (warm starts)
session = None

def handler(event, context):
    global session
    from rembg import remove, new_session
    import onnxruntime as ort

    # 1. Load pre-installed model with optimized ONNX settings
    model_path = "/var/task/models/u2net.onnx"

    if session is None:
        if os.path.exists(model_path):
            logger.info(f"Model found at {model_path}. Loading with optimized settings.")
            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = new_session("u2net", model_path=model_path, sess_opts=opts, providers=['CPUExecutionProvider'])
        else:
            logger.error(f"Model not found at {model_path}")
            contents = os.listdir('/var/task/models') if os.path.exists('/var/task/models') else 'Directory not found'
            logger.error(f"Contents of /var/task/models: {contents}")
            raise FileNotFoundError(f"Model not found at {model_path}")

    # 2. S3 event data
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        input_image = Image.open(io.BytesIO(response['Body'].read()))

        # 3. AI background removal
        logger.info(f"Starting AI processing for: {key}...")
        output_image = remove(input_image, session=session)

        # 4. Save as PNG and upload
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
