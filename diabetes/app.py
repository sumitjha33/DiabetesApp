from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the models and scaler
lr_model = joblib.load('logistic_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML form in index.html

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        data = request.form.to_dict()
        input_features = np.array([float(value) for value in data.values()]).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(input_features)

        # Predict using both models
        lr_prediction = lr_model.predict(scaled_features)[0]
        nb_prediction = nb_model.predict(scaled_features)[0]

        # Return results
        lr_result = "Positive" if lr_prediction == 1 else "Negative"
        nb_result = "Positive" if nb_prediction == 1 else "Negative"
        return render_template('result.html', lr_result=lr_result, nb_result=nb_result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
