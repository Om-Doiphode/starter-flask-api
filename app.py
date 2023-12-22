import os
import warnings
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
warnings.filterwarnings("ignore")

import json
from flask_cors import CORS
from flask import Flask, request, Response

import numpy as np
from PIL import Image
import requests

from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
cors = CORS(app)

global MODEL
global CLASSES


@app.route("/", methods=["GET"])
def default():
    return json.dumps({"Hello I am Chitti": "Speed 1 Terra Hertz, Memory 1 Zeta Byte"})


@app.route("/predict", methods=["GET"])
def predict():
    feature_extractor = AutoFeatureExtractor.from_pretrained('carbon225/vit-base-patch16-224-hentai')
    model = AutoModelForImageClassification.from_pretrained('carbon225/vit-base-patch16-224-hentai')
    src = request.args.get("src")
    print(f"{src=}")
    response = requests.get(src)
    print(f"{response=}")
    try:
        image = Image.open(BytesIO(response.content))
        image = image.resize((128, 128))
        image.save("new.jpg")
        encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        print(model.config.id2label[predicted_class_idx])
        # Return the predictions
        response=json.dumps({"class": model.config.id2label[predicted_class_idx]})
        response = Response(response)
        response.headers['Content-Security-Policy'] = "script-src 'self' render.com;"  # Modify this CSP as needed
        return response
    except:
        return json.dumps({"Uh oh": "We are down"})

