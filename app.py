from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("breast_cancer_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file)
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Malignant (Cancer Detected)"
    else:
        result = "Benign (No Cancer Detected)"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
