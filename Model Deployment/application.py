import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

# Initialize Flask app
application = Flask(__name__)
app = application

# Setup logging to track errors and activity
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load the model and scaler
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    print("Error loading model or scaler. Check logs.")

# Route for the main page
@app.route("/")
def index():
    return render_template('index.html')

# Route for the prediction functionality
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Retrieve form inputs
            Temperature = request.form.get('Temperature')
            RH = request.form.get('RH')
            Ws = request.form.get('WS')
            Rain = request.form.get('Rain')
            FFMC = request.form.get('FFMC')
            DMC = request.form.get('DMC')
            ISI = request.form.get('ISI')
            Classes = request.form.get('Classes')
            Region = request.form.get('Region')

            # Check if any inputs are missing
            if not all([Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]):
                return "Please provide all inputs.", 400

            # Convert inputs to float
            Temperature = float(Temperature)
            RH = float(RH)
            Ws = float(Ws)
            Rain = float(Rain)
            FFMC = float(FFMC)
            DMC = float(DMC)
            ISI = float(ISI)
            Classes = float(Classes)
            Region = float(Region)

            # Prepare data for scaling and prediction
            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

            # Scale the input data
            new_data_scaled = standard_scaler.transform(input_data)

            # Perform prediction using the model
            result = ridge_model.predict(new_data_scaled)

            # Return the result in the home.html
            return render_template('home.html', results=result[0])

        except ValueError as ve:
            return f"Invalid input: {ve}. Please make sure all fields contain valid numeric values.", 400
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return f"An error occurred during prediction: {e}", 500
    else:
        return render_template('home.html')

# Run the application
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)