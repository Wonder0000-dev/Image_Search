import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure image uploading
images = UploadSet("images", IMAGES)
app.config["UPLOADED_IMAGES_DEST"] = "uploads/"
configure_uploads(app, images)

# Load pre-trained model for image feature extraction (ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Folder containing local images to compare against
image_folder = 'local_images'

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Precompute features for all local images
image_features = {}
for filename in os.listdir(image_folder):
    if filename.endswith(('jpg', 'jpeg', 'png')):
        image_path = os.path.join(image_folder, filename)
        features = extract_features(image_path)
        image_features[filename] = features

# Route to upload the query image
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "image" in request.files:
        query_image = request.files["image"]
        query_image_path = os.path.join("uploads", query_image.filename)
        query_image.save(query_image_path)

        # Extract features for the query image
        query_features = extract_features(query_image_path)

        # Compare with all local images and find the most similar one
        similarities = {}
        for filename, features in image_features.items():
            similarity = cosine_similarity([query_features], [features])
            similarities[filename] = similarity[0][0]
        # Get the top 5 most similar images
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        print(sorted_images)

        return render_template("index.html", query_image=query_image.filename, sorted_images=sorted_images)

    return render_template("index.html")

# Route to serve uploaded images and local images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

@app.route("/local_images/<filename>")
def local_image(filename):
    return send_from_directory(image_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
