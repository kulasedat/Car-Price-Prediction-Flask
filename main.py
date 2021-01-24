from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('features.pkl', 'rb') as m:
    features = pickle.load(m)

with open('xgb_model_new', 'rb') as n:
    model = pickle.load(n)


@app.route('/')
def index():
    return "server is up and running"

@app.route('/predict', methods=['GET', 'POST'])

def predict():

    json_data = request.get_json()

    if not all(k in json_data for k in ["hp", "age", "km", "model"]):
        return "not enough data for the prediction"

    df = pd.DataFrame.from_dict([json_data])

    df = pd.get_dummies(df).reindex(columns=features, fill_value=0)

    prediction = model.predict(df)


    return str(prediction[0])



app.run()
