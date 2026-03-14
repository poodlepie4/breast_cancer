from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

# Create Flask app
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

    # Get uploaded image
    file = request.files["image"]

    # Open image
    img = Image.open(file)

    # Resize image for model
    img = img.resize((224, 224))

    # Convert to array
    img = np.array(img)

    # Normalize pixel values
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    # Convert prediction to label
    if prediction[0][0] > 0.5:
        result = "Malignant (Cancer Detected)"
    else:
        result = "Benign (No Cancer Detected)"

    # Send result to webpage
    return render_template("index.html", prediction=result)


# Run server
if __name__ == "__main__":
    app.run(debug=True)
