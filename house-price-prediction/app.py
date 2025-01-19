from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure you've saved the model previously using pickle)
with open('house-price-prediction/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the features
features = ['OverallQual', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'LotArea', 'YearBuilt']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        input_data = [float(request.form[feature]) for feature in features]

        # Create a DataFrame for the model
        input_df = pd.DataFrame([input_data], columns=features)

        # Make a prediction
        predicted_price = np.expm1(model.predict(input_df)[0])

        # Return the prediction to the user
        return render_template('index.html', prediction=f"Predicted Sale Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction="Error: Please check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)
