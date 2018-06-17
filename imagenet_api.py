import torch
from torchvision import models
from torch.autograd import Variable

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from imagenet_util import image_loader
import json

app = Flask(__name__)
cors = CORS(app)
model = models.vgg16(pretrained=True)
classes = dict(json.loads(open ('imagenet_classes.json').read()))

@app.route("/", methods=['POST'])
@cross_origin(origin='*')
def classify():
    image_file = request.files['image'].read()

    image = image_loader(image_file)

    prediction = model(image)

    prediction_class = classes[str(int(prediction.max(1)[1]))]

    return prediction_class

if __name__ == '__main__':
    app.run(debug=True)
