
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
data = pd.read_csv("IRIS.csv")

# Step 2: Separate input and output
X = data.drop("species", axis=1)
y = data["species"]

# Step 3: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the Machine Learning model
model = RandomForestClassifier(random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict flower species
predictions = model.predict(X_test)

# Step 7: Check model accuracy
accuracy = accuracy_score(y_test, predictions)

print("IRIS FLOWER CLASSIFICATION")
print("--------------------------")
print("Total dataset records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 8: Predict a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

result = model.predict(new_flower)

print("New flower measurements:", new_flower.values.tolist()[0])
print("Predicted flower species:", result[0])
