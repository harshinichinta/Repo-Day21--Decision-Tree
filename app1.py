import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit title
st.title("Decision Tree Regression - Car Dekho Dataset")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cardekho_data.csv")
    return df

# Read data
df = load_data()

st.subheader("Dataset")
st.dataframe(df.head())

# Preprocessing
st.subheader("Data Preprocessing")

# Create car age column
df['Car_Age'] = 2026 - df['Year']

# Drop unnecessary columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Convert categorical values to numerical
X = pd.get_dummies(X)

# Handle missing values
X = X.fillna(X.mean())

st.write("Processed Features:")
st.dataframe(X.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
st.subheader("Decision Tree Regressor Model")

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

st.success("Model Trained Successfully")

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
st.subheader("Model Evaluation")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")
st.write(f"R2 Score: {r2}")

# Hyperparameter tuning
st.subheader("Hyperparameter Tuning")

params = {
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'splitter': ['best', 'random']
}

grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

st.write("Best Parameters:")
st.write(grid_search.best_params_)

# Best model prediction
y_pred_best = best_model.predict(X_test)

st.write("Best Model R2 Score:")
st.write(r2_score(y_test, y_pred_best))

# Decision tree visualization
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=X.columns, ax=ax)
st.pyplot(fig)

# Actual vs Predicted
st.subheader("Actual vs Predicted")

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best
})

st.dataframe(comparison.head(10))

# Scatter plot
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred_best)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)

# User prediction section
st.subheader("Predict Car Selling Price")

present_price = st.number_input("Present Price", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=30000)
owner = st.selectbox("Owner", [0, 1, 2, 3])
car_age = st.number_input("Car Age", value=5)

fuel_petrol = st.selectbox("Fuel Type Petrol", [0, 1])
fuel_diesel = st.selectbox("Fuel Type Diesel", [0, 1])
seller_individual = st.selectbox("Seller Type Individual", [0, 1])
transmission_manual = st.selectbox("Transmission Manual", [0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Car_Age': [car_age],
        'Fuel_Type_Diesel': [fuel_diesel],
        'Fuel_Type_Petrol': [fuel_petrol],
        'Seller_Type_Individual': [seller_individual],
        'Transmission_Manual': [transmission_manual]
    })

    prediction = best_model.predict(input_data)

    st.success(f"Predicted Selling Price: {prediction[0]:.2f} Lakhs")

