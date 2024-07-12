#import predict #Uncomment this if we are not using flask and just 
# referencing the "predict" module from inside test.py
import requests

ride = {
    "PULocationID" : 10,
    "DOLocationID" : 50,
    "trip_distance" : 40
}

# We have moved the following lines of code to predict.py
# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())