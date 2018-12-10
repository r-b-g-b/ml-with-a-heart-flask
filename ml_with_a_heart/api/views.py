from ml_with_a_heart import app
from flask import request, jsonify
from ml_with_a_heart.api import utils
import json


class CustomException(Exception):
    """Exception for invalid JSON
    """
    status_code = 400

    def __init__(self, message, payload=None):
        super(CustomException, self).__init__(self)
        self.message = message
        self.payload = payload

    def to_dict(self):
        if self.payload is not None:
            response = self.payload
        else:
            response = dict()
        response['message'] = self.message
        return response



@app.route('/')
def home():
    return """<h1>Machine learning with a heart</h1>""".format()


@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease from posted features"""
    if request.method == 'POST':
        try:
            content = json.loads(request.get_json())
        except json.JSONDecodeError:
            raise CustomException("Invalid JSON", {'data': str(request.data)})

        data, missing_fields, extra_fields = utils.validate_content(content)
        if (len(missing_fields) > 0) or (len(extra_fields) > 0):
            raise CustomException("Fields do not match model fields",
                                  {'missing_fields': missing_fields, 'extra_fields': extra_fields})

        dataframe = utils.create_dataframe(data)

        try:
            features = utils.preprocess_data(dataframe)
        except ValueError as error_message:
            raise CustomException(f"Error transforming data. {error_message}")

        X = utils.concatenate_features(features)
        prediction, probability = utils.predict(X)

    return jsonify({'prediction': prediction.tolist(),
                    'probability': probability.tolist()})
