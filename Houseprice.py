import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pickle


# Load the dataset
train_data = pd.read_csv('train.csv')

# Preprocessing: Fill missing values
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        # Fill missing values with the mode for categorical features
        train_data[column] = train_data[column].fillna(train_data[column].mode()[0])
    else:
        # Fill missing values with the mean for numeric features
        train_data[column] = train_data[column].fillna(train_data[column].mean())

# Select features for the model
features = ['OverallQual','GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd','LotArea','YearBuilt']

# Create the feature matrix and target vector
X = train_data[features]
#y = train_data['SalePrice']
y = np.log1p(train_data['SalePrice'])


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)



# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

y_pred = model.predict(X_val)
r2_score = model.score(X_val, y_val)  # R-squared
mae = mean_absolute_error(y_val, y_pred)  # Mean Absolute Error
print(f"Model Accuracy (R²): {r2_score:.2f}")
print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")





# Function to get custom inputs and predict SalePrice
def predict_sale_price():
    print("Enter the following details to predict the SalePrice:")

    # Get user input for each feature
    OverallQual = float(input("Enter Overall Quality (OverallQual): "))
    GrLivArea = float(input("Enter Ground Living Area (GrLivArea): "))
    BedroomAbvGr = int(input("Enter number of Bedrooms Above Ground (BedroomAbvGr): "))
    FullBath = int(input("Enter number of Full Bathrooms (FullBath): "))
    HalfBath = int(input("Enter number of Half Bathrooms (HalfBath): "))
    TotRmsAbvGrd = int(input("Enter Total Rooms Above Ground (TotRmsAbvGrd): "))
    LotArea =  float(input("Enter Lot Area (LotArea): "))

    YearBuilt = int(input("Enter the year built (YearBuilt): "))

    # Create a DataFrame with the input values
    user_input = pd.DataFrame([[OverallQual,GrLivArea, BedroomAbvGr, FullBath, HalfBath, TotRmsAbvGrd,LotArea,YearBuilt]], columns=features)

    # Predict SalePrice
   # predicted_price = model.predict(user_input)[0]
    predicted_price = np.expm1(model.predict(user_input)[0])

    
    print(f"\nPredicted Sale Price: ₹{predicted_price:,.2f}")

# Call the function to make a prediction
predict_sale_price()
