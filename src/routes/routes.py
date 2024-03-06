from flask import Blueprint, request, jsonify
import io
from src.images.predict_image import predict_image_class

predict = Blueprint('predict', __name__)


@predict.route("/predict", methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_bytes = file.read()  # Read the content of the file stream
        img_stream = io.BytesIO(img_bytes)  # Create BytesIO object from file content
        predicted_class = predict_image_class(img_stream)  # Pass BytesIO object
        print(predicted_class)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})
