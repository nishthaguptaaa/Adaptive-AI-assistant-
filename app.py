import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# App Title
st.title("Adaptive AI Assistant for Data Science")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Run only if file uploaded
if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Dataset Information
    st.subheader("Dataset Information")

    st.write("Shape of Dataset:")
    st.write(df.shape)

    st.write("Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    missing = df.isnull().sum().sum()

    # Select Target Column
    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # Detect Problem Type
    if df[target].dtype == 'object':

        classification = True

        st.success(
            "Classification Problem Detected"
        )

    else:

        if df[target].nunique() <= 10:

            classification = True

            st.success(
                "Classification Problem Detected"
            )

        else:

            classification = False

            st.success(
                "Regression Problem Detected"
            )

    # Recommended Models
    st.subheader("Recommended Models")

    if classification:

        st.write("- Logistic Regression")
        st.write("- Decision Tree Classifier")
        st.write("- Random Forest Classifier")

    else:

        st.write("- Linear Regression")
        st.write("- Decision Tree Regressor")
        st.write("- Random Forest Regressor")

    # Features and Target
    X = df.drop(columns=[target])

    y = df[target]

    # Convert categorical feature columns
    for col in X.columns:

        if X[col].dtype == 'object':

            X[col] = LabelEncoder().fit_transform(
                X[col].astype(str)
            )

    # Convert target column
    if classification:

        y = LabelEncoder().fit_transform(
            y.astype(str)
        )

    # Fill missing values
    X = X.fillna(0)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Select Model
    if classification:

        model = LogisticRegression(
            max_iter=1000
        )

    else:

        model = LinearRegression()

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Model Performance
    st.subheader("Model Performance")

    if classification:

        predictions = [
            round(p) for p in predictions
        ]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.success(
            f"Model Accuracy: {accuracy:.2f}"
        )

    else:

        mse = mean_squared_error(
            y_test,
            predictions
        )

        st.success(
            f"Mean Squared Error: {mse:.2f}"
        )

    # AI Insights
    st.subheader("AI Insights")

    st.write(f"""
    - Dataset contains {missing} missing values
    - Problem Type Detected Automatically
    - Model trained successfully
    - Adaptive AI selected suitable workflow
    """)

    # Visualization
    st.subheader("Data Visualization")

    df.hist(figsize=(10, 8))

    st.pyplot(plt)