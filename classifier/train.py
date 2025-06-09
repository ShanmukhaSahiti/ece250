import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from utils import load_data

def main():
    # Load data and extract features based on the paper's methodology
    print("Loading data and extracting features per repetition...")
    X, y = load_data()
    
    if X.shape[0] == 0:
        print("No data was loaded. Please check the 'simulated_spectrograms' directory and the .mat files.")
        return

    print(f"Data loaded. Found {X.shape[0]} repetitions (samples) with {X.shape[1]} features each.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    # --- Dynamically handle SMOTE for small datasets ---
    # Find the number of samples in the smallest class
    min_class_count = np.min(np.bincount(y_train))
    
    # SMOTE requires k_neighbors < n_samples in the smallest class.
    # We only apply SMOTE if the smallest class has at least 2 samples.
    if min_class_count > 1:
        # Set k_neighbors to be one less than the number of samples in the smallest class
        k_neighbors = min_class_count - 1
        print(f"\nSmallest class has {min_class_count} samples. Setting SMOTE k_neighbors to {k_neighbors}.")

        # Apply SMOTE to the training data to handle class imbalance
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Shape of training data after SMOTE: {X_train.shape}")
    else:
        print(f"\nSkipping SMOTE because the smallest class has only {min_class_count} sample(s).")

    # Scale features AFTER resampling to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Logistic Regression (Linear) classifier
    print("\nTraining LogisticRegression classifier...")
    clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("Training complete.")
    
    # Evaluate the model
    print("\nEvaluating model performance on the original test set...")
    y_pred = clf.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Plot confusion matrix
    print("Displaying confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == '__main__':
    main() 