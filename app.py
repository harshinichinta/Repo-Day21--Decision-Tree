# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("Decision Tree Classifier Project")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ---------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("Dataset")
    st.write(df.head())

    # Dataset shape
    st.subheader("Dataset Shape")
    st.write(df.shape)

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Select target column
    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Convert categorical columns
    X = pd.get_dummies(X)

    # Fill missing values
    X = X.fillna(X.mean())

    # ---------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # HYPERPARAMETERS
    # ---------------------------------------------------

    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'splitter': ['best', 'random']
    }

    # ===================================================
    # GRID SEARCH CV
    # ===================================================

    st.header("Grid Search CV")

    grid_model = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=grid_model,
        param_grid=params,
        cv=5,
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, y_train)

    # Best model
    best_grid_model = grid_search.best_estimator_

    # Predictions
    grid_pred = best_grid_model.predict(X_test)

    # Metrics
    grid_accuracy = accuracy_score(y_test, grid_pred)

    grid_precision = precision_score(
        y_test,
        grid_pred,
        average='weighted'
    )

    grid_recall = recall_score(
        y_test,
        grid_pred,
        average='weighted'
    )

    grid_f1 = f1_score(
        y_test,
        grid_pred,
        average='weighted'
    )

    # Results
    st.subheader("Best Parameters - Grid Search")
    st.write(grid_search.best_params_)

    st.subheader("Accuracy")
    st.write(grid_accuracy)

    st.subheader("Precision")
    st.write(grid_precision)

    st.subheader("Recall")
    st.write(grid_recall)

    st.subheader("F1 Score")
    st.write(grid_f1)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, grid_pred))

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, grid_pred))

    # ===================================================
    # RANDOMIZED SEARCH CV
    # ===================================================

    st.header("Randomized Search CV")

    random_model = DecisionTreeClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=random_model,
        param_distributions=params,
        n_iter=10,
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    random_search.fit(X_train, y_train)

    # Best model
    best_random_model = random_search.best_estimator_

    # Predictions
    random_pred = best_random_model.predict(X_test)

    # Metrics
    random_accuracy = accuracy_score(y_test, random_pred)

    random_precision = precision_score(
        y_test,
        random_pred,
        average='weighted'
    )

    random_recall = recall_score(
        y_test,
        random_pred,
        average='weighted'
    )

    random_f1 = f1_score(
        y_test,
        random_pred,
        average='weighted'
    )

    # Results
    st.subheader("Best Parameters - Randomized Search")
    st.write(random_search.best_params_)

    st.subheader("Accuracy")
    st.write(random_accuracy)

    st.subheader("Precision")
    st.write(random_precision)

    st.subheader("Recall")
    st.write(random_recall)

    st.subheader("F1 Score")
    st.write(random_f1)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, random_pred))

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, random_pred))

    # ===================================================
    # DECISION TREE VISUALIZATION
    # ===================================================

    st.header("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(25, 15))

    plot_tree(
        best_grid_model,
        feature_names=X.columns,
        class_names=[str(c) for c in best_grid_model.classes_],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    st.pyplot(fig)

else:
    st.info("Please upload a CSV dataset.")