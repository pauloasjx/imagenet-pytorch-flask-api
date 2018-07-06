
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from imagenet_util import image_loader, model, classes

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=['GET'])
def index():
	return send_from_directory('.', 'index.html')

@app.route("/classify", methods=['POST'])
@cross_origin(origin='*')
def classify():
	image = request.files['image'].read()
	image = image_loader(image)

	prediction = model(image)

	prediction_class = classes[str(int(prediction.max(1)[1]))]

	return jsonify(
		prediction=prediction_class[1]
	)

if __name__ == '__main__':
    app.run(debug=True)
