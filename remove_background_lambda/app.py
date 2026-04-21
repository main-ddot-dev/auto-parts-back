import os
import boto3
import io
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ort_session = None

def handler(event, context):
    global ort_session

    if ort_session is None:
        task_root = os.environ.get('LAMBDA_TASK_ROOT', '/var/task')
        model_path = os.path.join(task_root, 'models', 'u2net.onnx')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        ort_session = ort.InferenceSession(model_path, sess_options=sess_opts)
        logger.info(f"Model loaded from: {model_path}")

    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        img = Image.open(io.BytesIO(response['Body'].read())).convert("RGB")
        original_size = img.size

        img_resized = img.resize((320, 320), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized).astype('float32') / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        input_data = np.transpose(img_np, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        logger.info(f"Processing: {key}")
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        mask = ort_session.run(None, ort_inputs)[0][0][0]

        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = np.where(mask < 0.2, 0.0, mask)

        mask_img = Image.fromarray((mask * 255).astype('uint8')).resize(original_size, Image.Resampling.LANCZOS)
        img.putalpha(mask_img)

        out_buffer = io.BytesIO()
        img.save(out_buffer, format='PNG')
        out_buffer.seek(0)

        dest_bucket = os.environ['DESTINATION_BUCKET']
        dest_prefix = os.environ['DESTINATION_PREFIX']
        new_key = f"{dest_prefix}{os.path.basename(key).rsplit('.', 1)[0]}.png"

        s3.put_object(Bucket=dest_bucket, Key=new_key, Body=out_buffer, ContentType='image/png')
        logger.info(f"Success: image saved to {new_key}")

        return {"statusCode": 200, "body": f"Success: {key} processed and saved to {new_key}"}

    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}", exc_info=True)
        raise e
