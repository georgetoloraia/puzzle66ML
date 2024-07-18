import secrets
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

import keras
from keras.layers import Dense
from keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import joblib

def generate_random_private_key():
    return secrets.token_hex(32)

def compute_rmd160(private_key):
    sha256_hash = hashlib.sha256(bytes.fromhex(private_key)).digest()
    rmd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    return rmd160_hash.hex()

def create_dataset(num_samples):
    data = {'Private Key': [], 'RIPEMD-160': []}
    
    for _ in range(num_samples):
        private_key = generate_random_private_key()
        rmd160_hash = compute_rmd160(private_key)
        data['Private Key'].append(private_key)
        data['RIPEMD-160'].append(rmd160_hash)
    
    return pd.DataFrame(data)

def preprocess_data(df):
    df['Private Key (int)'] = df['Private Key'].apply(lambda x: int(x, 16))
    df['RIPEMD-160 (int)'] = df['RIPEMD-160'].apply(lambda x: int(x, 16))
    return df

def split_data(df):
    X = df[['Private Key (int)']]
    y = df['RIPEMD-160 (int)']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train):
    model = build_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

def predict_rmd160(model, private_key):
    private_key_int = int(private_key, 16)
    private_key_df = pd.DataFrame({'Private Key (int)': [private_key_int]})
    predicted_rmd160 = model.predict(private_key_df)
    return hex(int(predicted_rmd160[0]))[2:].zfill(40)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def save_model(model, file_name):
    model.save(file_name)

def load_model(file_name):
    return tf.keras.models.load_model(file_name)

def main():
    # Create dataset
    df = create_dataset(1000)
    df = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Predict the RIPEMD-160 hash for a random private key
    test_private_key = generate_random_private_key()
    predicted_rmd160 = predict_rmd160(model, test_private_key)
    print(f"Test Private Key: {test_private_key}")
    print(f"Predicted RIPEMD-160 Hash: {predicted_rmd160}")
    
    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    
    # Save the model
    save_model(model, 'neural_network_rmd160_model.h5')
    
    # (Optional) Load the model if needed
    # model = load_model('neural_network_rmd160_model.h5')

if __name__ == '__main__':
    main()
