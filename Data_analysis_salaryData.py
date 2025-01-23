import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Function to train the model
def train_model(df):
    # Split the data into features and target
    X = df[['YearsExperience']]  # Feature
    y = df['Salary']  # Target

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to predict salary based on years of experience
def predict_salary(model, experience):
    input_data = pd.DataFrame([[experience]], columns=['YearsExperience'])
    predicted_salary = model.predict(input_data)
    return predicted_salary[0]

# Streamlit Web App
def main():
    st.title("Salary Predictor Based on Years of Experience")
    
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Check if the columns 'YearsExperience' and 'Salary' exist
        if 'YearsExperience' in df.columns and 'Salary' in df.columns:
            # Train the model with the uploaded dataset
            model = train_model(df)
            
            # Get user input for years of experience
            experience_input = st.number_input("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

            # Check if the input is within the dataset's range
            min_experience = df['YearsExperience'].min()
            max_experience = df['YearsExperience'].max()

            if experience_input < min_experience or experience_input > max_experience:
                st.warning(f"You've entered a value of {experience_input} years, which is outside the range of the dataset ({min_experience} to {max_experience} years). The model will still provide a prediction based on the regression line.")

            # Predict the salary
            predicted_salary = predict_salary(model, experience_input)
                
            # Display the predicted salary
            st.write(f"Predicted Salary for {experience_input} years of experience is: ${predicted_salary:,.2f}")
                
            # Plotting the data and regression line
            fig, ax = plt.subplots(figsize=(8, 6))
                
            # Scatter plot of the actual data
            ax.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Actual Data')
                
            # Generate the regression line
            x_range = np.linspace(df['YearsExperience'].min(), df['YearsExperience'].max(), 100).reshape(-1, 1)
            y_range = model.predict(x_range)
                
            # Plot the regression line
            ax.plot(x_range, y_range, color='red', label='Regression Line')
                
            # Highlight the predicted point on the regression line
            ax.scatter(experience_input, predicted_salary, color='green', label=f"Prediction at {experience_input} years", zorder=5)
                
            # Labels and title
            ax.set_xlabel('Years of Experience')
            ax.set_ylabel('Salary')
            ax.set_title('Linear Regression - Years of Experience vs Salary')
            ax.legend()
                
            # Show the plot
            st.pyplot(fig)

        else:
            st.error("The uploaded dataset must contain 'YearsExperience' and 'Salary' columns.")

if __name__ == "__main__":
    main()
