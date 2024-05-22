from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import load_model, save_model, preprocess_data

app = Flask(__name__)

# Endpoint to train the model
@app.route('/train', methods=['POST'])
def train_model():
    # Load and preprocess the dataset
    data = request.get_json()
    df = pd.DataFrame(data)
    
    df, X, y = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    save_model(model, 'model.pkl')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return jsonify({'message': 'Model trained and saved', 'accuracy': accuracy})

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = load_model('classifier_model.pkl')
    
    # Get the data from the request
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Preprocess the data
    df, X, _ = preprocess_data(df, training=False)
    
    # Make predictions
    predictions = model.predict(X)
    
    return jsonify({'predictions': predictions.tolist()})

@app.get('/')
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
