import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('Housing.csv')

# Encode categorical variable
df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])

# Create binary label
df['label'] = (df['price'] >= 5000000).astype(int)

# Features and label
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'furnishingstatus']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Evaluate
y_pred = log_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(log_model, 'flat_price_classifier.pkl')
print("✅ Logistic regression model saved as flat_price_classifier.pkl")

# Load the saved model
model = joblib.load("flat_price_classifier.pkl")

# Sample input as a DataFrame with column names
sample_data = pd.DataFrame([{
    'area': 4520,
    'bedrooms': 2,
    'bathrooms': 2,
    'stories': 1,
    'parking':1,
    'furnishingstatus': 1  # Furnished (assuming 1 = furnished)
}])

# Predict
prediction = model.predict(sample_data)
print("Prediction:", "Expensive" if prediction[0] == 1 else "Affordable") 
