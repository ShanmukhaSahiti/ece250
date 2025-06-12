import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
from sklearn.preprocessing import StandardScaler

def train_classifier():
    # Get the absolute path to the spectrograms directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'simulated_spectrograms')
    
    # Load and preprocess the data
    print("Loading all data...")
    X, y = load_data(data_dir)
    print(f"Data loaded: {len(X)} samples.")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balance the data with SMOTE
    print("\nBalancing data with SMOTE...")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    # Train the classifier on the entire dataset
    print("\nTraining model on full dataset...")
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_balanced, y_balanced)
    
    # Evaluate the classifier on the same data it was trained on
    print("\nEvaluating model on training data...")
    y_pred = clf.predict(X_balanced)
    accuracy = accuracy_score(y_balanced, y_pred)
    
    print(f"\nAccuracy on Full Training Dataset: {accuracy:.4f}\n")
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_balanced, y_pred))

    # Plot confusion matrix
    print("\nDisplaying confusion matrix...")
    cm = confusion_matrix(y_balanced, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.classes_,
                yticklabels=clf.classes_)
    plt.title('Confusion Matrix on Full Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    train_classifier() 