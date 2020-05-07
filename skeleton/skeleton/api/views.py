import cv2
import numpy as np
from flask import request, current_app, jsonify

_LABELS = ['dog', 'cat']


def ping():
    return 'Ping'


def classify():
    f = request.files['image'].read()
    npimg = np.fromstring(f, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    prediction = current_app.model.predict(img)
    predicted_class = np.argmax(prediction)
    result = {'class': _LABELS[predicted_class], 'probability': float(prediction[predicted_class])}
    return jsonify(result)
