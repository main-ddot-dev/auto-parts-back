import os
import shutil
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
        # 1. Copy model to writable /tmp if not already there
        model_home = "/tmp/.u2net"
        os.environ['U2NET_HOME'] = model_home
        if not os.path.exists(model_home):
            logger.info("Copying model to /tmp for write permissions...")
            shutil.copytree("/var/task/.u2net", model_home)

        # Lazy load rembg to avoid Lambda init timeout (10s limit)
        from rembg import remove

        # 2. Which file was just uploaded to S3?
        source_bucket = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']

        # Save it to the output bucket, changing the extension to PNG
        destination_bucket = os.environ.get('DESTINATION_BUCKET', 'piezas-sin-fondo')
        destination_prefix = os.environ.get('DESTINATION_PREFIX', 'sin-fondo/')
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        destination_key = destination_prefix + base_name + '.png'

        logger.info(f"Starting AI processing for: {file_key}...")

        # 3. Download the image directly into Lambda's RAM
        response = s3.get_object(Bucket=source_bucket, Key=file_key)
        image_bytes = response['Body'].read()
        original_image = Image.open(io.BytesIO(image_bytes))

        # 4. AI background removal
        clean_image = remove(original_image)

        # 5. Prepare the final PNG in memory
        buffer = io.BytesIO()
        clean_image.save(buffer, format="PNG")
        buffer.seek(0)

        # 6. Upload to the destination bucket
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
