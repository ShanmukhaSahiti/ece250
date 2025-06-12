import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
    print("Loading data and extracting features...")
    X, y = load_data(data_dir)
    
    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    print("\nStarting 5-fold cross-validation...")
    
    # Initialize the scaler
    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/5 ---")
        
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Print class distribution before SMOTE
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print(f"Class distribution before SMOTE: {class_counts}")
        
        # Apply SMOTE to balance the training data
        # Ensure k_neighbors is less than the number of samples in the smallest class
        min_class_count = np.min(class_counts)
        k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Print class distribution after SMOTE
        unique_classes, class_counts = np.unique(y_train_balanced, return_counts=True)
        print(f"Class distribution after SMOTE: {class_counts}")
        
        # Train the classifier
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on test set
        accuracy = clf.score(X_test_scaled, y_test)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    
    # Print cross-validation summary
    print("\n--- Cross-Validation Summary ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    # Train final model on all data (scaled)
    print("\nTraining final model on all data...")
    X_scaled = scaler.fit_transform(X)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    final_clf = LogisticRegression(max_iter=2000, random_state=42)
    final_clf.fit(X_balanced, y_balanced)
    
    # Get predictions for all data
    y_pred = final_clf.predict(X_scaled)
    
    # Print classification report
    print("\nOverall Classification Report (from all folds):")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    print("\nDisplaying overall confusion matrix...")
    cm = confusion_matrix(y, y_pred, labels=final_clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=final_clf.classes_,
                yticklabels=final_clf.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    train_classifier() 