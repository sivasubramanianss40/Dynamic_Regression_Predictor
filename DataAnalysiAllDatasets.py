import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to handle categorical data encoding
def encode_categorical_data(df):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # If it's categorical, we perform Label Encoding (you can also use OneHotEncoder)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Function to train the model
def train_model(df, x_column, y_column):
    # Drop any unwanted columns like "Unnamed: 0" if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Encode categorical columns to numerical
    df = encode_categorical_data(df)
    
    # Split the data into features and target
    X = df[[x_column]]  # Feature (independent variable)
    y = df[y_column]  # Target (dependent variable)

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to predict y based on the input X value
def predict_value(model, x_value, x_column):
    # Ensure that input data has the same column name as the model was trained on
    input_data = pd.DataFrame([[x_value]], columns=[x_column])  # Use the exact column name used during training
    predicted_value = model.predict(input_data)
    return predicted_value[0]

# Streamlit Web App
def main():
    st.set_page_config(page_title="Dynamic Regression Predictor", layout="centered")
    
    st.title("Dynamic Regression Predictor")
    
    # Display some text for guidance
    st.write("""
    ### Instructions
    1. Upload your dataset (CSV format).
    2. Select the independent (X) and dependent (y) variables.
    3. Enter the value for the independent variable to predict the dependent value.
    4. View the prediction and regression visualization below.
    """)
    
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Drop any unwanted columns (e.g., 'Unnamed: 0' or any other irrelevant columns)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Show the first few rows of the dataframe to understand its structure
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Allow the user to choose the independent (X) and dependent (y) variables
        st.subheader("Select Columns")
        x_column = st.selectbox("Select Independent Variable (X)", df.columns)
        y_column = st.selectbox("Select Dependent Variable (y)", df.columns)

        # Check if the selected column for X contains non-numeric data
        if df[x_column].dtype == 'object':
            st.warning(f"The selected independent variable '{x_column}' is categorical and will be encoded.")

        # Train the model with the selected columns
        model = train_model(df, x_column, y_column)

        # Get user input for the independent variable (X)
        st.subheader(f"Enter the value for {x_column}")
        
        # Check if the X column is numeric or categorical
        if df[x_column].dtype in ['int64', 'float64']:
            x_input = st.number_input(f"Enter {x_column} value", min_value=float(df[x_column].min()), 
                                     max_value=float(df[x_column].max()), step=0.1)
        else:
            # If the column is categorical, ask for selection from the available options
            unique_values = df[x_column].unique()
            x_input = st.selectbox(f"Select {x_column} value", unique_values)

        # Check if the input is within the dataset's range
        min_value = df[x_column].min()
        max_value = df[x_column].max()

        if isinstance(x_input, (int, float)) and (x_input < min_value or x_input > max_value):
            st.warning(f"You've entered a value of {x_input}, which is outside the range of the dataset ({min_value} to {max_value}). The model will still provide a prediction based on the regression line.")

        # Predict the dependent variable (y) for the input X value
        predicted_value = predict_value(model, x_input, x_column)

        # Display the predicted value
        st.subheader("Prediction Result")
        st.write(f"Predicted {y_column} for {x_column} = {x_input} is: **{predicted_value:.2f}**")

        # Plotting the data and regression line
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot of the actual data
        ax.scatter(df[x_column], df[y_column], color='blue', label='Actual Data')

        # Generate the regression line
        x_range = np.linspace(df[x_column].min(), df[x_column].max(), 100).reshape(-1, 1)
        y_range = model.predict(x_range)

        # Plot the regression line
        ax.plot(x_range, y_range, color='red', label='Regression Line')

        # Highlight the predicted point on the regression line
        ax.scatter(x_input, predicted_value, color='green', label=f"Prediction at {x_input}", zorder=5)

        # Labels and title
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"Linear Regression: {x_column} vs {y_column}")
        ax.legend()

        # Show the plot
        st.pyplot(fig)

        st.write("""
        ### Additional Information
        The plot above shows the actual data points (blue), the regression line (red), 
        and the predicted value (green) based on your input for the independent variable.
        """)

        # Calculate and display the mean of x_column and y_column only if they are numeric
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            mean_x = df[x_column].mean()
            mean_y = df[y_column].mean()

            st.write(f"### Mean of {x_column} and {y_column}")
            st.write(f"Mean of {x_column}: **{mean_x:.2f}**")
            st.write(f"Mean of {y_column}: **{mean_y:.2f}**")
        else:
            st.warning(f"Cannot calculate the mean for non-numeric columns: '{x_column}' and '{y_column}'.")

if __name__ == "__main__":
    main()
