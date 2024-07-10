import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.
    
    Parameters:
    - file_path: str, the path to the CSV file.
    
    Returns:
    - DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Convert the private keys from hex to integers.
    
    Parameters:
    - df: DataFrame, the input data.
    
    Returns:
    - DataFrame with an additional column for the integer private keys.
    """
    df['Private Key (int)'] = df['Private Key'].apply(lambda x: int(x, 16))
    return df

def split_data(df):
    """
    Split the data into training and testing sets.
    
    Parameters:
    - df: DataFrame, the input data.
    
    Returns:
    - X_train, X_test, y_train, y_test: DataFrames and Series for training and testing.
    """
    X = df[['Bit Range']]
    y = df['Private Key (int)']
    return train_test_split(X, y, test_size=0.001, random_state=42)

def train_model(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    
    Returns:
    - model: the trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_private_key(model, bit_range):
    """
    Predict the private key for a given bit range using the trained model.
    
    Parameters:
    - model: the trained Linear Regression model.
    - bit_range: int, the bit range to predict for.
    
    Returns:
    - predicted_private_key_hex: str, the predicted private key in hex format.
    """
    bit_range_df = pd.DataFrame({'Bit Range': [bit_range]})
    predicted_private_key = model.predict(bit_range_df)
    return hex(int(predicted_private_key[0]))[2:].zfill(64)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error.
    
    Parameters:
    - model: the trained Linear Regression model.
    - X_test: DataFrame, testing features.
    - y_test: Series, testing labels.
    
    Returns:
    - mse: float, the Mean Squared Error.
    """
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def save_model(model, file_name):
    """
    Save the trained model to a file.
    
    Parameters:
    - model: the trained Linear Regression model.
    - file_name: str, the file name to save the model to.
    """
    joblib.dump(model, file_name)

def load_model(file_name):
    """
    Load a trained model from a file.
    
    Parameters:
    - file_name: str, the file name to load the model from.
    
    Returns:
    - model: the loaded Linear Regression model.
    """
    return joblib.load(file_name)

def main():
    # Load and preprocess data
    df = load_data('alldata.csv')
    df = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Predict the private key for Bit Range 66
    predicted_private_key_66_hex = predict_private_key(model, 66)
    print(f"Predicted Private Key for Bit Range 66: {predicted_private_key_66_hex}")
    
    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    
    # Save the model
    save_model(model, 'linear_regression_model.pkl')
    
    # (Optional) Load the model if needed
    # model = load_model('linear_regression_model.pkl')

if __name__ == '__main__':
    main()
