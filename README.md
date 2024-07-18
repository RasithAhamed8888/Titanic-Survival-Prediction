# Titanic-Survival-Prediction


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Data Exploration
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.describe())

# Data Cleaning
# Drop irrelevant columns
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Convert categorical columns to numerical
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

# Data Visualization
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Prepare data for training
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Predict on new data (example)
new_data = pd.DataFrame({
    'Pclass': [3, 1],
    'Age': [22, 38],
    'SibSp': [1, 1],
    'Parch': [0, 0],
    'Fare': [7.25, 71.2833],
    'Sex_male': [1, 0],
    'Embarked_Q': [0, 0],
    'Embarked_S': [1, 0]
})

new_data = scaler.transform(new_data)
predictions = model.predict(new_data)
print('Predictions for new data:', predictions)
