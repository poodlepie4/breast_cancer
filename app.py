from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("breast_cancer_model.h5")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", prediction="No image selected")

    # Open image
    img = Image.open(file)

    # Resize for model
    img = img.resize((224,224))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Malignant (Cancer Detected)"
    else:
        result = "Benign (No Cancer Detected)"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
