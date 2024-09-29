from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Result page - processes form and shows prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from all features
        features = [
            'has_fever', 'has_coughing_blood', 'has_sputum_blood',
            'has_night_sweats', 'has_chest_pain', 'has_back_pain',
            'has_shortness_of_breath', 'has_weight_loss', 'has_body_fatigue',
            'has_lumps', 'has_continuous_cough', 'has_swollen_lymph_nodes',
            'has_loss_of_appetite'
        ]

        # Collect form responses and map them to binary (1 for Yes, 0 for No)
        input_data = [int(request.form[feature]) for feature in features]

        # Convert input to numpy array
        input_data = np.array(input_data).reshape(1, -1)

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Predict using the KNN model
        prediction = model.predict(scaled_data)

        # Map prediction to a readable result
        result = 'Positive for Tuberculosis' if prediction[0] else 'Negative for Tuberculosis'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
