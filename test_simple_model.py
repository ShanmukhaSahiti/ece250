import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_spectrogram(spectrogram, target_shape=(100, 1000)):
    """Normalize spectrogram to target shape using interpolation"""
    current_shape = spectrogram.shape
    zoom_factors = (target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])
    return zoom(spectrogram, zoom_factors, order=1)

def load_spectrogram_data(data_dir):
    """Load spectrogram data from .mat files"""
    X = []
    y = []
    class_mapping = {
        'lateral lunge': 0,
        'sit up': 1,
        'stiff-leg deadlift': 2
    }
    
    # Load data from each activity folder
    for activity in class_mapping.keys():
        activity_dir = os.path.join(data_dir, activity)
        if not os.path.exists(activity_dir):
            logging.warning(f"Directory not found: {activity_dir}")
            continue
            
        logging.info(f"Processing activity: {activity}")
        for file in os.listdir(activity_dir):
            if file.endswith('.mat'):
                file_path = os.path.join(activity_dir, file)
                try:
                    logging.info(f"Loading file: {file_path}")
                    data = sio.loadmat(file_path)
                    
                    # Print all keys in the .mat file
                    logging.info(f"Keys in {file}: {data.keys()}")
                    
                    # Get spectrograms from all links
                    sp_all = data['sp_all']
                    logging.info(f"Found {len(sp_all)} links in {file}")
                    logging.info(f"Type of sp_all: {type(sp_all)}")
                    
                    # Process each link's spectrogram
                    for link_idx in range(len(sp_all)):
                        # Extract the actual spectrogram matrix from the object array
                        spectrogram = sp_all[link_idx][0]
                        logging.info(f"Link {link_idx + 1} spectrogram shape: {spectrogram.shape}")
                        logging.info(f"Link {link_idx + 1} spectrogram type: {spectrogram.dtype}")
                        
                        # Normalize spectrogram to common shape
                        normalized_spectrogram = normalize_spectrogram(spectrogram)
                        logging.info(f"Normalized shape: {normalized_spectrogram.shape}")
                        logging.info(f"Link {link_idx + 1} spectrogram min: {np.min(normalized_spectrogram)}, max: {np.max(normalized_spectrogram)}")
                        
                        # Flatten the spectrogram
                        X.append(normalized_spectrogram.flatten())
                        y.append(class_mapping[activity])
                        
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue
    
    if len(X) == 0:
        raise ValueError("No data was loaded. Check the data directory and file structure.")
    
    X = np.array(X)
    y = np.array(y)
    logging.info(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
    logging.info(f"Input shape: {X.shape}")
    logging.info(f"X data type: {X.dtype}")
    logging.info(f"X min: {np.min(X)}, max: {np.max(X)}")
    return X, y

def main():
    try:
        # Load the data
        logging.info("Loading spectrogram data...")
        X, y = load_spectrogram_data('simulated_spectrograms')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train a Random Forest classifier
        logging.info("Training Random Forest classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        logging.info("Making predictions...")
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test Accuracy: {accuracy:.2f}")
        
        # Print detailed classification report
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_test, y_pred, 
                                        target_names=['lateral lunge', 'sit up', 'stiff-leg deadlift']))
        
        # Print feature importance
        feature_importance = clf.feature_importances_
        logging.info(f"Number of important features (importance > 0.01): {np.sum(feature_importance > 0.01)}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main() 