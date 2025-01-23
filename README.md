# Dynamic Regression Predictor - Streamlit Web Application

This is a Streamlit-based web application that allows users to upload a CSV dataset, perform linear regression analysis, and make predictions. The app helps users select independent (X) and dependent (Y) variables, train a regression model, and visualize the regression results.

## Features

- **Dataset Upload:** Upload your own CSV file for regression analysis.
- **Data Preview:** Preview the first few rows of the dataset for better understanding.
- **Variable Selection:** Select independent (X) and dependent (Y) variables for the regression model.
- **Automatic Categorical Data Handling:** The app automatically detects and encodes categorical variables using Label Encoding.
- **Prediction:** Input a value for the independent variable (X) to predict the corresponding dependent variable (Y).
- **Visualization:** View a scatter plot of the data, the regression line, and the predicted value for the selected input.
- **Statistical Insights:** Displays the mean of both the independent and dependent variables if they are numeric.

## Usage Instructions

1. **Upload Dataset:** Upload your CSV dataset using the file uploader widget.
2. **Select Variables:** Choose the independent (X) and dependent (Y) variables from the dropdown menus.
3. **Enter Value:** Enter a value for the independent variable (X) to predict the dependent variable (Y).
4. **View Prediction and Plot:** View the predicted result for the dependent variable and a regression plot with the data points, regression line, and predicted value.

## Tools & Libraries Used

- **Streamlit**: For creating the interactive web application interface.
- **Pandas**: For data manipulation and processing.
- **Scikit-learn**: For building and training the linear regression model.
- **Matplotlib**: For plotting the data and regression line.
- **Numpy**: For numerical operations such as generating a range for plotting the regression line.
- **LabelEncoder (from scikit-learn)**: For encoding categorical variables to numeric values.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/dynamic-regression-predictor.git
   cd dynamic-regression-predictor
