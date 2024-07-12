import os
import pickle

import mlflow
from flask import Flask, request, jsonify


# RUN_ID = os.getenv('RUN_ID')
RUN_ID = "00bf5a395e7c4f1ba527d172918b5b4a"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-duration")

# If we log the model to an S3 bucket, then we do not have to worry
# about the ML flow tracking server being down and can use the 
# logged model to get the predictions even when the tracking server is offline
#logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)  