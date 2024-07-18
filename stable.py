import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrivateKeyPredictor:
    def __init__(self, model_file=None):
        self.model = None
        if model_file:
            self.load_model(model_file)
        self.known_ranges = {
            66: (int('20000000000000000', 16), int('3ffffffffffffffff', 16))
        }

    def load_data(self, file_path):
        """
        Load data from a CSV file and return a DataFrame.
        
        Parameters:
        - file_path: str, the path to the CSV file.
        
        Returns:
        - DataFrame containing the data.
        """
        try:
            df = pd.read_csv(file_path)
            logging.info("Data loaded successfully.")
            return df
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e

    def preprocess_data(self, df):
        """
        Convert the private keys from hex to integers and validate data.
        
        Parameters:
        - df: DataFrame, the input data.
        
        Returns:
        - DataFrame with an additional column for the integer private keys.
        """
        try:
            df['Private Key (int)'] = df['Private Key'].apply(lambda x: int(x, 16))
            logging.info("Data preprocessed successfully.")
            return df
        except ValueError as e:
            logging.error(f"Error converting hex to int: {e}")
            raise e

    def split_data(self, df, test_size=1):
        """
        Split the data into training and testing sets.
        
        Parameters:
        - df: DataFrame, the input data.
        - test_size: float, the proportion of the dataset to include in the test split.
        
        Returns:
        - X_train, X_test, y_train, y_test: DataFrames and Series for training and testing.
        """
        X = df[['Bit Range']]
        y = df['Private Key (int)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logging.info("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_type='linear'):
        """
        Train a model.
        
        Parameters:
        - X_train: DataFrame, training features.
        - y_train: Series, training labels.
        - model_type: str, type of model to train ('linear' or 'random_forest').
        
        Returns:
        - model: the trained model.
        """
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
        else:
            logging.error("Unsupported model type.")
            return None
        
        model.fit(X_train, y_train)
        logging.info(f"{model_type.capitalize()} model trained successfully.")
        self.model = model
        return model

    def predict_private_key(self, bit_range):
        """
        Predict the private key for a given bit range using the trained model.
        
        Parameters:
        - bit_range: int, the bit range to predict for.
        
        Returns:
        - predicted_private_key_hex: str, the predicted private key in hex format.
        """
        if self.model is None:
            logging.error("Model is not trained or loaded.")
            return None
        
        bit_range_df = pd.DataFrame({'Bit Range': [bit_range]})
        predicted_private_key = self.model.predict(bit_range_df)[0]
        
        if bit_range in self.known_ranges:
            min_key, max_key = self.known_ranges[bit_range]
            predicted_private_key = max(min(predicted_private_key, max_key), min_key)
        
        predicted_private_key_hex = hex(int(predicted_private_key))[2:].zfill(64)
        logging.info(f"Predicted Private Key for Bit Range {bit_range}: {predicted_private_key_hex}")
        return predicted_private_key_hex

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model using Mean Squared Error.
        
        Parameters:
        - X_test: DataFrame, testing features.
        - y_test: Series, testing labels.
        
        Returns:
        - mse: float, the Mean Squared Error.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model evaluation completed. Mean Squared Error: {mse}")
        return mse

    def save_model(self, file_name):
        """
        Save the trained model to a file.
        
        Parameters:
        - file_name: str, the file name to save the model to.
        """
        joblib.dump(self.model, file_name)
        logging.info(f"Model saved to {file_name}.")

    def load_model(self, file_name):
        """
        Load a trained model from a file.
        
        Parameters:
        - file_name: str, the file name to load the model from.
        
        Returns:
        - model: the loaded model.
        """
        self.model = joblib.load(file_name)
        logging.info(f"Model loaded from {file_name}.")

def main():
    predictor = PrivateKeyPredictor()

    # Load and preprocess data
    df = predictor.load_data('alldata.csv')
    df = predictor.preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = predictor.split_data(df)
    
    # Train the model
    predictor.train_model(X_train, y_train, model_type='random_forest')
    
    # Predict the private key for Bit Range 66
    predicted_private_key_66_hex = predictor.predict_private_key(66)
    print(f"Predicted Private Key for Bit Range 66: {predicted_private_key_66_hex}")
    
    # Evaluate the model
    mse = predictor.evaluate_model(X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    
    # Save the model
    predictor.save_model('random_forest_model.pkl')
    
    # (Optional) Load the model if needed
    # predictor.load_model('random_forest_model.pkl')

if __name__ == '__main__':
    main()
