from flask import Flask, request, render_template, jsonify, send_file
import coremltools as ct
from PIL import Image
import json
import cv2
import numpy as np
import os

app = Flask(__name__)

model = ct.models.MLModel("./static/Model60k.mlmodel")
model_s = ct.models.MLModel("./static/Model16k.mlmodel")

@app.route("/transform", methods=['POST'])
def transform():
    if request.method == 'POST':
        token = request.form['token']
        if token != "0c8HLwy59MuA9QOnp9JCBQ":
            return "Invalid token"
        f = request.files['image']
        if f.filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            return "Invalid file type"
        f.save(f.filename)
        image = Image.open(f.filename)
        image = image.resize((360, 360)) \
                     .convert('RGB')
        image.save(f.filename)
        return send_file(f.filename, mimetype='image/'+f.filename.split(".")[-1])

def detect_abstract_shapes(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    contours_labels = []
    if len(contours) > 2:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    for contour in contours:
        if i == 0:
            i = 1
            continue
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        # get convex hull
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        permiter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (permiter * permiter)
        if circularity > 0.7:
            return "smile"
        elif len(approx) >= 3 and len(approx) < 7:
            return "angry"

    return ",".join(contours_labels)


def fmt_prediction(prediction):
    target = prediction["target"]
    confidence = prediction["targetProbability"][target]
    return target

def get_confidence(prediction):
    target = prediction["target"]
    confidence = prediction["targetProbability"][target]
    return confidence

@app.route("/draw")
def draw():
    return render_template("canvas.html")

def use_model(model, image, as_dict=False):
    prediction = model.predict({"image": image})
    confidence = get_confidence(prediction)
    prediction = fmt_prediction(prediction)
    if as_dict:
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    return f"{prediction} ({confidence * 100:.1f}%)"

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        token = request.form['token']
        if token != "0c8HLwy59MuA9QOnp9JCBQ":
            return "Invalid token"
        f = request.files['image']
        if f.filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            return "Invalid file type"
        f.filename = "temp." + str(os.urandom(8).hex()) + "." + f.filename.split(".")[-1]
        f.save(f.filename)
        image = Image.open(f.filename)
        image = image.resize((360, 360)) \
                     .convert('RGB')
        image.save(f.filename)
        shape = detect_abstract_shapes(f.filename)
        if "json" in request.form:
            return jsonify({
                "model60k": use_model(model, image, as_dict=True),
                "model16k": use_model(model_s, image, as_dict=True),
                "cv2": {
                    "prediction": shape,
                    "confidence": None
                }
            })
        return f"""
        Model 60K Response: {use_model(model, image)},
        Model 16K Response: {use_model(model_s, image)},
        CV2 Response: {shape}
        """
    else:
        return "Invalid request"

@app.route("/")
def index():
    return """
    <h1>CoreML Flask App</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="hidden" name="token" value="0c8HLwy59MuA9QOnp9JCBQ">
        <input type="hidden" name="json" value="true">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Submit">
    </form>
    <h2>Get transform result</h2>
    <form action="/transform" method="post" enctype="multipart/form-data">
        <input type="hidden" name="token" value="0c8HLwy59MuA9QOnp9JCBQ">
        <input type="hidden" name="json" value="true">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Submit">
    </form>
    """

if __name__ == "__main__":
    app.run(debug=True)