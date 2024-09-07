from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import joblib
import re
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})  # Enable CORS for /predict route

# Load the PCA model
pca_model = joblib.load('pca_model.joblib')

# Load the scaler
scaler = StandardScaler(with_mean=False, with_std=False)

# Initialize the PorterStemmer
stemmer = PorterStemmer()
def preprocess_input(input_data):
    # Text Cleaning
    cleaned_data = re.sub(r'[^a-zA-Z\s]', '', input_data)
    
    # Tokenization
    tokens = cleaned_data.split()
    
    # Lowercasing
    tokens_lower = [token.lower() for token in tokens]
    
    # Stemming
    tokens_stemmed = [stemmer.stem(token) for token in tokens_lower]
    
    # Convert tokens to ASCII values
    numerical_data = []
    for token in tokens_stemmed:
        if len(token) > 1:
            numerical_data.append(ord(token[0]))
        else:
            numerical_data.append(ord(token))
    
    # Ensure input data has 10 features
    while len(numerical_data) < 10:
        numerical_data.extend(numerical_data)  # Repeat the data
    return numerical_data[:10]  # Trim to 10 features

def count_vowels(input_data):
    vowels = 'aeiouAEIOU'
    vowel_count = sum(input_data.count(vowel) for vowel in vowels)
    vowel_frequency = {char.lower(): input_data.lower().count(char.lower()) for char in input_data if char.lower() in vowels}
    return vowel_count, vowel_frequency
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json['input_data']  # Assuming input data is in JSON format

    # Print the received data
    print("Received raw input data:", input_data)

    # Preprocess the input data
    preprocessed_input = preprocess_input(input_data)

    # Ensure that the input data has 10 features
    while len(preprocessed_input) < 10:
        preprocessed_input.append(0)  # Pad with zeros if necessary

    # Reshape the input data to a 2D array
    preprocessed_input = np.array(preprocessed_input).reshape(1, -1)

    # Make predictions using the loaded PCA model
    predicted_probs = pca_model.predict_proba(preprocessed_input)[:, 1]  # Predicted probabilities for class 1

    # Convert predicted probabilities to predicted class labels (using a threshold of 0.5)
    predicted_class = (predicted_probs >= 0.5).astype(int)

    # Count vowels and calculate their frequency
    vowel_count, vowel_frequency = count_vowels(input_data)

    # Define the message based on the predicted class
    if predicted_class[0] == 1:
        message = "The speech has vowels."
    else:
        message = "The speech does not have vowels."

    # Construct the response
    response = {
        'predicted_class': predicted_class.tolist(),  # Convert to list for JSON serialization
        'message': message,
        'vowel_count': vowel_count,
        'vowel_frequency': vowel_frequency
    }

    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True)
