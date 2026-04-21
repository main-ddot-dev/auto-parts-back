import os
import io
import logging
import numpy as np
import scipy.ndimage as ndimage
import onnxruntime as ort
import boto3
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ort_session = None

def handler(event, context):
    global ort_session

    if ort_session is None:
        model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', '/var/task'), 'models', 'isnet.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        ort_session = ort.InferenceSession(model_path, sess_options=sess_opts)
        logger.info(f"ISNet loaded from: {model_path}")

    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        img = Image.open(io.BytesIO(response['Body'].read())).convert("RGB")
        original_size = img.size

        img_resized = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized).astype('float32') / 255.0
        img_np = (img_np - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_data = np.expand_dims(np.transpose(img_np, (2, 0, 1)), axis=0)

        logger.info(f"Processing: {key}")
        raw_mask = ort_session.run(None, {ort_session.get_inputs()[0].name: input_data})[0][0][0]
        logger.info(
            f"Mask — min:{raw_mask.min():.3f} max:{raw_mask.max():.3f} mean:{raw_mask.mean():.3f} "
            f"p10:{np.percentile(raw_mask, 10):.3f} p50:{np.percentile(raw_mask, 50):.3f} p90:{np.percentile(raw_mask, 90):.3f}"
        )

        mask = (raw_mask - raw_mask.min()) / (raw_mask.max() - raw_mask.min())
        mask = ndimage.gaussian_filter(mask, sigma=1.5)
        binary_mask = ndimage.binary_fill_holes(mask > 0.4)
        mask = ndimage.gaussian_filter(binary_mask.astype(np.float32), sigma=1.5)

        mask_img = Image.fromarray((mask * 255).astype('uint8')).resize(original_size, Image.Resampling.LANCZOS)
        img.putalpha(mask_img)

        out_buffer = io.BytesIO()
        img.save(out_buffer, format='PNG')
        out_buffer.seek(0)

        dest_bucket = os.environ['DESTINATION_BUCKET']
        dest_prefix = os.environ['DESTINATION_PREFIX']
        new_key = f"{dest_prefix}{os.path.basename(key).rsplit('.', 1)[0]}.png"

        s3.put_object(Bucket=dest_bucket, Key=new_key, Body=out_buffer, ContentType='image/png')
        logger.info(f"Saved: {new_key}")

        return {"statusCode": 200, "body": f"Success: {key} → {new_key}"}

    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}", exc_info=True)
        raise e
