import os

# Force U2NET_HOME to /tmp BEFORE any rembg import
# This prevents "Read-only file system" errors on /home/sbx_user1051
os.environ['U2NET_HOME'] = '/tmp'

import boto3
import io
import logging
from PIL import Image

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS S3 client
s3 = boto3.client('s3')

def handler(event, context):
    try:
        # 1. Symlink baked model from /var/task to writable /tmp
        baked_model = '/var/task/model_data/.u2net/u2net.onnx'
        tmp_model_dir = '/tmp/.u2net'
        tmp_model_file = '/tmp/.u2net/u2net.onnx'

        if not os.path.exists(tmp_model_file):
            os.makedirs(tmp_model_dir, exist_ok=True)
            try:
                os.symlink(baked_model, tmp_model_file)
                logger.info("Symlink created: AI reads from internal storage.")
            except FileExistsError:
                pass

        # 2. Import rembg AFTER model path is ready
        from rembg import remove, new_session

        logger.info("Loading AI session...")
        session = new_session("u2net")

        # 3. Which file was just uploaded to S3?
        source_bucket = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']

        # Save it to the output bucket, changing the extension to PNG
        destination_bucket = os.environ.get('DESTINATION_BUCKET', 'piezas-sin-fondo')
        destination_prefix = os.environ.get('DESTINATION_PREFIX', 'sin-fondo/')
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        destination_key = destination_prefix + base_name + '.png'

        logger.info(f"Starting AI processing for: {file_key}...")

        # 4. Download the image directly into Lambda's RAM
        response = s3.get_object(Bucket=source_bucket, Key=file_key)
        image_bytes = response['Body'].read()
        original_image = Image.open(io.BytesIO(image_bytes))

        # 5. AI background removal
        clean_image = remove(original_image, session=session)

        # 6. Prepare the final PNG in memory
        buffer = io.BytesIO()
        clean_image.save(buffer, format="PNG")
        buffer.seek(0)

        # 7. Upload to the destination bucket
        s3.put_object(
            Bucket=destination_bucket,
            Key=destination_key,
            Body=buffer,
            ContentType='image/png'
        )

        logger.info(f"Success: {file_key} -> {destination_bucket}/{destination_key}")

        return {
            "statusCode": 200,
            "body": f"Success: {file_key} processed and saved to {destination_key}"
        }

    except Exception as e:
        logger.error(f"Error processing the image: {str(e)}", exc_info=True)
        raise e
