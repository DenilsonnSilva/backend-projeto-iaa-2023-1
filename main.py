from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
model = load_model("path/to/your/model.h5")

if __name__ == "__main__":
    app.run()
