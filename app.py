from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model and scaler
# For simplicity, we simulate the model and scaler loading here
# You should replace these with the actual loading from your files
model = RandomForestClassifier(random_state=42)
scaler = StandardScaler()

# Assume we have trained the model and scaler on the dataset
def train_model():
    # Load the dataset
    df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
    df = df.drop(columns=['patientid'])
    
    # Prepare data
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Fit scaler
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Train the model
    model.fit(X_scaled, y)

train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    chestpain = int(request.form['chestpain'])
    restingBP = int(request.form['restingBP'])
    serumcholestrol = int(request.form['serumcholestrol'])
    fastingbloodsugar = int(request.form['fastingbloodsugar'])
    restingrelectro = int(request.form['restingrelectro'])
    maxheartrate = int(request.form['maxheartrate'])
    exerciseangia = int(request.form['exerciseangia'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    noofmajorvessels = int(request.form['noofmajorvessels'])
    
    # Create a numpy array with the input values
    new_individual = np.array([[age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, 
                                restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels]])
    
    # Scale the input data
    new_individual_scaled = scaler.transform(new_individual)
    
    # Predict using the trained model
    prediction = model.predict(new_individual_scaled)
    
    # Interpret the result
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    
    return render_template('result.html', risk=risk)

if __name__ == "__main__":
    app.run(debug=True)
