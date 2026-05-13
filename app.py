# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Title
st.title("Decision Tree Classifier")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(df.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Convert categorical columns to numeric
    X = pd.get_dummies(X)

    # Handle missing values
    X = X.fillna(X.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'splitter': ['best', 'random']
    }

    model = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Best Parameters")
    st.write(grid_search.best_params_)

    st.subheader("Accuracy")
    st.write(f"{accuracy:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Plot Decision Tree
    st.subheader("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(
        best_model,
        filled=True,
        feature_names=X.columns,
        class_names=[str(i) for i in np.unique(y)],
        ax=ax
    )

    st.pyplot(fig)

else:
    st.info("Please upload a CSV file.")