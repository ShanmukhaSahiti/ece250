import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network model
class ActivityRecognitionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ActivityRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_spectrogram_data(data_dir):
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
                    
                    # Get spectrograms from all links
                    sp_all = data['sp_all']
                    logging.info(f"Found {len(sp_all)} links in {file}")
                    
                    # Process each link's spectrogram
                    for link_idx in range(len(sp_all)):
                        spectrogram = sp_all[link_idx]
                        # Flatten the spectrogram
                        X.append(spectrogram.flatten())
                        y.append(class_mapping[activity])
                        logging.info(f"Processed link {link_idx + 1}, shape: {spectrogram.shape}")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue
    
    if len(X) == 0:
        raise ValueError("No data was loaded. Check the data directory and file structure.")
    
    X = np.array(X)
    y = np.array(y)
    logging.info(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
    logging.info(f"Input shape: {X.shape}")
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

def main():
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load and preprocess data
        logging.info("Loading spectrogram data...")
        X, y = load_spectrogram_data('simulated_spectrograms')
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        logging.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Create data loaders
        train_dataset = SpectrogramDataset(X_train, y_train)
        val_dataset = SpectrogramDataset(X_val, y_val)
        test_dataset = SpectrogramDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model, loss function, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y))
        logging.info(f"Input size: {input_size}, Number of classes: {num_classes}")
        
        model = ActivityRecognitionModel(input_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        logging.info("Starting model training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, device=device
        )
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('training_losses.png')
        
        # Evaluate on test set
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        logging.info(f'Test Accuracy: {test_accuracy:.2f}%')
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main() 