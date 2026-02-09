from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Model aur Scaler load karein (Ensure filenames are correct)
model = load_model('churn_model.h5')
scaler = pickle.load(open('scaler1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # User inputs according to your sequence
    credit_score = float(request.form['CreditScore'])
    geography = request.form['Geography']
    gender = request.form['Gender']
    age = float(request.form['Age'])
    tenure = float(request.form['Tenure'])
    balance = float(request.form['Balance'])
    num_products = float(request.form['NumOfProducts'])
    has_card = float(request.form['HasCrCard'])
    active_member = float(request.form['IsActiveMember'])
    salary = float(request.form['EstimatedSalary'])

    # Internal Encoding (Model needs these specific columns)
    geo_germany = 1 if geography == 'Germany' else 0
    geo_spain = 1 if geography == 'Spain' else 0
    gender_male = 1 if gender == 'Male' else 0

    # Model features order (training order): 
    # [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geo_Ger, Geo_Spa, Gen_Male]
    features = [credit_score, age, tenure, balance, num_products, has_card, active_member, salary, geo_germany, geo_spain, gender_male]

    # Scaling and Prediction
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    
    result = "EXIT (Churn)" if prediction[0][0] > 0.5 else "STAY (No Churn)"
    color = "#9fc5d6" if prediction[0][0] > 0.5 else "#f1cae8" # Red if churn, Green if stay
    prob = round(float(prediction[0][0]) * 100, 2)

    return render_template('index.html', 
                           prediction_text=result,
                           probability=f'{prob}%',
                           res_color=color)

if __name__ == "__main__":
    app.run(debug=True)