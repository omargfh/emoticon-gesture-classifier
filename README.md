# Emoticon Gesture Classifier
Classifier for hand-drawn shapes using CoreML and OpenCV

# Image Classification
Image classification was done using **CoreML**. It uses data sourced from [Google Quick! Draw](https://quickdraw.withgoogle.com/data) for hand-drawn `28 x 28` shapes (ear, line, circle, triangle, cloud, zigzag). We train two independent models on 16k and 60k training points respectively.

## Converting from `*.npy` to `.png`
```python
import numpy as np
import PIL.Image as Image
# Convert numpy bitmap to image and save it
def bitmap_to_image(bitmap, filename):
    # Create image from bitmap
    image = Image.fromarray(bitmap)
    # Save image
    image.save(filename)
    return

def convert_np_arr_to_images(npfilename, outname, path="./img", range=None):
    f = np.load(npfilename)

    for idx, i in enumerate(f):
        if not (
            range and idx >= range[0] and idx < range[1]
        ):
            continue
        mkdir(f'{path}/{outname}')
        bitmap_to_image(
            i.reshape(28, 28),
            f'{path}/{outname}/{outname}'+str(idx)+'.png'
        )

names_to_classes = {
    './np/full_numpy_bitmap_circle.npy': 'smile',
    './np/full_numpy_bitmap_triangle.npy': 'angry',
    './np/full_numpy_bitmap_line.npy': 'like',
    './np/full_numpy_bitmap_cloud.npy': 'cry',
    './np/full_numpy_bitmap_zigzag.npy': 'haha',
    './np/full_numpy_bitmap_ear.npy': 'heart'
}
for filepath, classname in names_to_classes.items():
    convert_np_arr_to_images(
        filepath, classname,
        path="./large_model/target",
        range=(0,10_000)
    )
    convert_np_arr_to_images(
        filepath, classname,
        path="./large_model/test",
        range=(11_000,12_000)
    )
```

## Other attempted models in `TensorFlow`
```python
def train_model(data):
    # Define a mapping from string labels to integer values
    label_to_int = {label: idx for idx, label in enumerate(data.keys())}

    # Convert the string labels to integer labels in your data
    X = []
    y = []

    for label, images in data.items():
        label_int = label_to_int[label]  # Convert label to integer
        X.extend(images)
        y.extend([label_int] * len(images) )

    num_classes = len(data.keys())
    X = np.array(X)
    y = np.array(y, dtype=np.uint8)  # Ensure labels are of integer data type

    print(
        'X shape:', X.shape,
        'y shape:', y.shape
    )

    # Define and compile the model
    model = tf.keras.models.Sequential([
        # reshape (784, ) into (28, 28)
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.Conv2D(128, 3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(256, 3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=3)

    return model
```

# Web API
To utilize `CoreML`, we create an API endpoint to connect it to our app using `Flask`.
```python
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
        os.remove(f.filename)
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
```

#### Canvas.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            min-width: 100vw;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        canvas{ border: 1px solid black; }
        button {
            padding: 10px 20px;
            border: 1px solid black;
            border-radius: 5px;
            background-color: #fff;
            cursor: pointer;
        }
        button:hover {
            background-color: #eee;
        }
        span {
            margin: 10px;
            font-size: 20px;
            font-family: sans-serif;
        }
    </style>
</head>
<body>
    <canvas width="1200" height="1200" style="scale: 0.3; position: absolute; transform: translateY(-500px);" id="canvas"></canvas>
    <span id="shape" style="margin-top: 200px;">No shape</span>
    <button id="new">New shape</button>
    <script>
        const $newButton = document.querySelector("#new");
        const $shapeLabel = document.querySelector("#shape");

        const $canvas = document.querySelector("#canvas");
        const ctx = $canvas.getContext("2d");
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, $canvas.width, $canvas.height);

        $blankImage = ctx.getImageData(0, 0, $canvas.width, $canvas.height);
        $canvas.addEventListener("mousedown", (e) => {
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
            $canvas.addEventListener("mousemove", draw);
        });
        $canvas.addEventListener("mouseup", (e) => {
            $canvas.removeEventListener("mousemove", draw);
            ctx.stroke();
            makePrediction();
        });
        function draw(e) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 15;
            ctx.lineCap = "round";
            ctx.miterLimit = 3;
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
        }
        async function makePrediction() {
            const image = ctx.getImageData(0, 0, $canvas.width, $canvas.height);
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append("token", "0c8HLwy59MuA9QOnp9JCBQ");
                formData.append("image", blob, "image.png");
                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    })
                    const data = await response.text();
                    $shapeLabel.innerHTML = data.split(",").join("<br/>")
                } catch {
                    alert("Something went wrong")
                }
            })
        }

        $newButton.addEventListener("click", () => {
            ctx.putImageData($blankImage, 0, 0);
            $shapeLabel.textContent = "No shape";
        });
    </script>
</body>
</html>
```

### CV Shape Identification
We also use CV to verify results for basic abstract shapes (circle, triangle).

### Classifier Results
![Classifier Results](./docs/image.png)
