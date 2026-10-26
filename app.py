
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("IRIS.csv")

# Prepare data
X = data.drop("species", axis=1)
y = data["species"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Website title
st.title("Iris Flower Classification")
st.write("Enter flower measurements to predict its species.")

st.subheader("Flower Measurements")

# User inputs
sepal_length = st.number_input(
    "Sepal Length (cm)", 0.0, 10.0, 5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)", 0.0, 10.0, 3.5
)
petal_length = st.number_input(
    "Petal Length (cm)", 0.0, 10.0, 1.4
)
petal_width = st.number_input(
    "Petal Width (cm)", 0.0, 10.0, 0.2
)

# Prediction button
if st.button("Predict Flower Species"):
    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )

    prediction = model.predict(flower)[0]

    st.success(f"Predicted Species: {prediction}")

    st.subheader("Flower Details")
    st.write("Sepal Length:", sepal_length, "cm")
    st.write("Sepal Width:", sepal_width, "cm")
    st.write("Petal Length:", petal_length, "cm")
    st.write("Petal Width:", petal_width, "cm")

st.divider()
st.caption("Machine Learning project | CodSoft Data Science Internship")
