import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, load_data_from_files

def train_classifier():
    # Get the absolute path to the spectrograms directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'simulated_spectrograms')
    
    # --- 1. Split data into training and test sets (one file per activity for testing) ---
    print("--- Splitting data into training and test sets ---")
    train_files = []
    test_files = []
    activity_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for activity in activity_folders:
        activity_path = os.path.join(data_dir, activity)
        mat_files = [os.path.join(activity_path, f) for f in os.listdir(activity_path) if f.endswith('.mat')]
        
        if mat_files:
            # Sort files to ensure consistent splitting
            mat_files.sort()
            # Add the first file to the test set and the rest to the training set
            test_files.append(mat_files[0])
            train_files.extend(mat_files[1:])
            print(f"Activity '{activity}': {len(mat_files)-1} train, 1 test")
            
    # --- 2. Load training data ---
    print("\n--- Loading training data ---")
    X_train_full, y_train_full = load_data_from_files(train_files)
    
    if X_train_full.size == 0:
        print("\nNo training data was loaded. Exiting.")
        return

    print(f"\nTraining data loaded: {len(X_train_full)} samples.")
    
    # --- 3. K-Fold Cross-Validation on the Training Set ---
    print("\n--- Starting 5-fold cross-validation on the training set ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        print(f"\n--- Fold {fold}/5 ---")
        
        # Split the training data into training and validation for this fold
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Apply SMOTE to balance the fold's training data
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = np.min(class_counts)
        k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # Train the classifier
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on the validation set for this fold
        accuracy = clf.score(X_val_scaled, y_val)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")

    # Print cross-validation summary
    print("\n--- Cross-Validation Summary on Training Set ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    # --- 4. Train a final model on the full training set ---
    print("\n--- Training the final model on all training data ---")
    
    # Scale the full training set
    scaler_final = StandardScaler()
    X_train_full_scaled = scaler_final.fit_transform(X_train_full)
    
    # Balance the full training set
    unique_classes, class_counts = np.unique(y_train_full, return_counts=True)
    min_class_count = np.min(class_counts)
    k_neighbors_final = min(5, min_class_count - 1) if min_class_count > 1 else 1
    smote_final = SMOTE(random_state=42, k_neighbors=k_neighbors_final)
    X_train_balanced_final, y_train_balanced_final = smote_final.fit_resample(X_train_full_scaled, y_train_full)

    # Train the final classifier
    final_clf = LogisticRegression(max_iter=2000, random_state=42)
    final_clf.fit(X_train_balanced_final, y_train_balanced_final)
    print("Model training complete.")

    # --- NEW: Evaluate on the training data itself ---
    y_train_pred = final_clf.predict(X_train_balanced_final)
    train_accuracy = accuracy_score(y_train_balanced_final, y_train_pred)
    print(f"\nAccuracy on Training Set: {train_accuracy:.4f}")

    # --- 5. Evaluate the final model on the unseen test set ---
    print("\n--- Loading test data and evaluating final model ---")
    X_test, y_test = load_data_from_files(test_files)

    if X_test.size == 0:
        print("\nNo test data was loaded. Cannot evaluate.")
        return

    # Scale the test data using the scaler from the *full* training set
    X_test_scaled = scaler_final.transform(X_test)
    
    # Get predictions for the test set
    y_pred = final_clf.predict(X_test_scaled)
    
    # --- 6. Report results ---
    print("\n--- Test Results on Unseen Files ---")
    for i, file_path in enumerate(test_files):
        print(f"File: {os.path.basename(file_path):<20} | True: {y_test[i]:<20} | Predicted: {y_pred[i]:<20}")
        
    # Overall accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy on Unseen Test Set: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report for Unseen Test Set:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Plot confusion matrix for the test set
    print("\nDisplaying confusion matrix for the test set...")
    cm = confusion_matrix(y_test, y_pred, labels=final_clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=final_clf.classes_,
                yticklabels=final_clf.classes_)
    plt.title('Confusion Matrix for Unseen Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    train_classifier() 