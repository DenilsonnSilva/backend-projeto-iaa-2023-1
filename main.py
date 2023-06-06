from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
model = load_model("path/to/your/model.h5")

if __name__ == "__main__":
    app.run()

def preprocess(img_array):
    img_resized = np.array(Image.fromarray(img_array).resize((224, 224)))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def get_class_info(prediction):
    classes = {
        0: ("class_id_0", "Carro"),
        1: ("class_id_1", "Moto"),
        2: ("class_id_2", "Caminhão"),
    }
    return classes[prediction]