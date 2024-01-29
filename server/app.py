from flask import Flask, request, jsonify
from flask_cors import CORS
from server import from_jpg_toTensor, get_bbs, get_text_bb
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import requests
import torch
import json
import cv2
from matplotlib import pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/api', methods=["POST"])
def predict():
    raw_data = request.data.decode("utf-8")
    json_data = json.loads(raw_data)
    image_url = json_data['url']
    if "jpeg" in image_url:
        image_data = image_url.replace("data:image/jpeg;base64,", "")
    elif "png" in image_url:
        image_data = image_url.replace("data:image/png;base64,", "")

    # Decode base64 to binary data
    binary_data = base64.b64decode(image_data)

    # Convert binary data to NumPy array
    image_array = np.array(Image.open(BytesIO(binary_data)))
    if image_array.shape[-1] == 4:  # Check if the image has an alpha channel
        image_array = image_array[:, :, :3]
    try:
        img_tensor = torch.tensor(np.array(image_array))
        img_tensor = img_tensor.permute(2, 0, 1)

        predictions, bbs = get_bbs(img_tensor)

        text_bbs = get_text_bb(img_tensor, bbs[0])
        text_detection_img = predictions.numpy()
        text_detection_img = (text_detection_img - text_detection_img.min()) / (text_detection_img.max() - text_detection_img.min())
        text_detection_img = (text_detection_img * 255).astype(np.uint8)
        text_detection_img = text_detection_img.reshape(img_tensor.shape[1], img_tensor.shape[2])
        # image = cv2.imread('frame.png', 0)
        colormap = plt.get_cmap('viridis')
        text_detection_img = (colormap(text_detection_img) * 2**10).astype(np.uint8)[:,:,:3]
        text_detection_img = np.array(cv2.cvtColor(text_detection_img, cv2.COLOR_RGB2BGR))

        image = Image.fromarray(text_detection_img)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        data_url = f"data:image/png;base64,{image_data}"

        return jsonify({"text_bbs": text_bbs, "data_url": data_url})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
