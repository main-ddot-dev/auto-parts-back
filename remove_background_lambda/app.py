import os
import boto3
import io
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global ONNX session for warm starts
ort_session = None

def handler(event, context):
    global ort_session

    # 1. Load model (only on cold start)
    if ort_session is None:
        logger.info("Loading lightweight ONNX AI engine...")
        model_path = "/var/task/models/u2net.onnx"
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        ort_session = ort.InferenceSession(model_path, sess_options=sess_opts)

    # 2. S3 event data
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # 3. Download and prepare image
        response = s3.get_object(Bucket=bucket, Key=key)
        img = Image.open(io.BytesIO(response['Body'].read())).convert("RGB")
        original_size = img.size

        # Resize for u2net (expects 320x320)
        input_img = img.resize((320, 320), Image.Resampling.LANCZOS)
        input_data = np.array(input_img).astype('float32') / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        # 4. Run AI inference (1-2 seconds)
        logger.info(f"Processing: {key}")
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)

        # 5. Post-process: normalize mask and apply as alpha channel
        mask = ort_outs[0][0][0]
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = Image.fromarray((mask * 255).astype('uint8')).resize(original_size, Image.Resampling.LANCZOS)
        img.putalpha(mask)

        # 6. Save as PNG and upload
        out_buffer = io.BytesIO()
        img.save(out_buffer, format='PNG')
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
