from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model("model.h5")


def preprocess(img_array):
    img_resized = np.array(Image.fromarray(img_array).resize((224, 224)))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def get_class_info(prediction):
    classes = {
        0: ("0", "Carro"),
        1: ("1", "Moto"),
        2: ("2", "Caminhão"),
    }
    return classes[prediction]


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Não foi encontrada nenhuma imagem"})

    file = request.files["image"]
    img = Image.open(file.stream)
    img_array = np.array(img)
    img_processed = preprocess(img_array)
    prediction = model.predict(img_processed)
    class_id, class_name = get_class_info(np.argmax(prediction))
    response = {"class_id": class_id, "class_name": class_name}
    return jsonify(response)


@app.route("/")
def home():
    return jsonify({"title": "Hello World!"})


@app.route("/teste", methods=["POST"])
def test():
    data = request.get_json()
    message = data.get("message")
    response = {"result": "success", "message": message}
    return jsonify(response)


if __name__ == "__main__":
    app.run()
