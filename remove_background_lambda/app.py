import os
import boto3
import io
import logging
from PIL import Image

# --- CRITICAL: must run BEFORE any rembg/numba import ---
# Disable Numba cache entirely (avoids "no locator" / read-only errors)
os.environ['NUMBA_DISABLE_CACHE'] = '1'
# Force Lambda-compatible threading
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
os.environ['OMP_NUM_THREADS'] = '1'
# Prevent rembg from writing to /home/sbx_user1051
os.environ['U2NET_HOME'] = '/tmp'

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    # 1. Symlink baked model to /tmp/u2net.onnx
    baked_model = "/var/task/models/u2net.onnx"
    target_model = "/tmp/u2net.onnx"

    if not os.path.exists(target_model):
        try:
            if os.path.exists(baked_model):
                os.symlink(baked_model, target_model)
                logger.info(f"Symlink created: {target_model} -> {baked_model}")
        except Exception as e:
            logger.warning(f"Error creating symlink: {e}")

    # 2. Lazy import and session init
    from rembg import remove, new_session

    logger.info("Initializing AI session...")
    session = new_session("u2net", model_path=target_model)

    # 3. S3 event data
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
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
