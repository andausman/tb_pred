import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv('data/tubdata.csv')

# Drop irrelevant columns
df = df.drop(columns=['no', 'name', 'gender'])

# Binary Features
binary_features = [
    'fever for two weeks', 'coughing blood', 'sputum mixed with blood',
    'night sweats', 'chest pain', 'back pain in certain parts',
    'shortness of breath', 'weight loss', 'body feels tired',
    'lumps that appear around the armpits and neck',
    'cough and phlegm continuously for two weeks to four weeks',
    'swollen lymph nodes', 'loss of appetite'
]

for feature in binary_features:
    df[f'has_{feature.replace(" ", "_")}'] = df[feature].apply(lambda x: 1 if x > 0 else 0)

# Create target column
df['has_tuberculosis'] = (df[[f'has_{feature.replace(" ", "_")}' for feature in binary_features]].sum(axis=1) >= 7).astype(int)

# Define features and target
X = df[[f'has_{feature.replace(" ", "_")}' for feature in binary_features]]
y = df['has_tuberculosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of training data
print("Shape of training data:", X_train.shape)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved!")