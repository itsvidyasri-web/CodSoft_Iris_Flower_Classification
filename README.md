# Iris Flower Classification Using Machine Learning

## Project Overview
This project predicts the species of an Iris flower using its sepal and petal measurements. It uses a Machine Learning model and an interactive Streamlit web application.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Machine Learning

## Dataset
The dataset contains 150 flower records with four input features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target column is `species`, which contains the flower species.

## Model
A Random Forest Classifier is used to train the model and predict flower species.

The dataset is split into 80% training data and 20% testing data for evaluation.

## Features
- Interactive web interface
- User input for flower measurements
- Real-time species prediction
- Model accuracy and classification report in the Python script

## How to Run
1. Install the required packages:
   `python -m pip install pandas scikit-learn streamlit`

2. Run the application:
   `python -m streamlit run app.py`

3. Enter the flower measurements and click **Predict Flower Species**.

## Project Result
The model achieved 100% accuracy on the current test split. This result applies to that particular split and does not guarantee perfect predictions on all new data.

## Internship
CodSoft Data Science Internship

## Author
Vidya Sri K
