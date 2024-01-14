import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import os


class Original(nn.Module):
    def __init__(self, num_classes):
        super(Original, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm5 = nn.BatchNorm1d(1024)
        self.batchnorm6 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.batchnorm1(self.pool(nn.functional.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool(nn.functional.relu(self.conv2(x))))
        x = self.batchnorm3(self.pool(nn.functional.relu(self.conv3(x))))
        x = self.batchnorm4(self.pool(nn.functional.relu(self.conv4(x))))
        x = x.view(-1, 256 * 6 * 6)
        x = self.batchnorm5(self.dropout(nn.functional.relu(self.fc1(x))))
        x = self.batchnorm6(self.dropout(nn.functional.relu(self.fc2(x))))
        x = self.fc3(x)
        return x



class OriginalModel():
    def __init__(self, dataset_path, train_size, batch_size, device, num_epochs):
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.num_classes = os.listdir(self.dataset_path).__len__() - 1 if '.DS_Store' in os.listdir(self.dataset_path) else os.listdir(self.dataset_path).__len__()
        self.classes = [x for x in os.listdir(self.dataset_path) if x != '.DS_Store']

    def transform(self):
        transform = transforms.Compose([
                    transforms.Resize((100, 100)), # Resize to 100x100
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for each color channel
                ])
        return transform

    def upload_dataset(self):
        
        dataset = ImageFolder(self.dataset_path, transform=self.transform())

        train_size = int(self.train_size * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
    
    def train(self):
        train_loader, test_loader = self.upload_dataset()

        self.model = Original(self.num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        best_val_accuracy = 0.0

        # Training Loop
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over the training data
            for i, data in enumerate(train_loader, 0):
                # Get the inputs and labels
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate and print training statistics
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_loss /= len(test_loader)
                val_accuracy = 100 * correct / total
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), 'models/original_model.pth')
                    print(f'Saved new best model with accuracy: {best_val_accuracy:.2f}%')

        return self.model


    def predict(self, image):
        # Load the trained model
        original_model = Original(self.num_classes).to(self.device)
        original_model.load_state_dict(torch.load('models/original_model.pth', map_location=self.device))
        original_model.eval()

        image = Image.fromarray(image)
        # Define the transform
        transform = self.transform()

        # Apply the transformations
        image = transform(image)

        # Add an extra batch dimension since pytorch treats all inputs as batches
        image = image.unsqueeze(0).to(self.device)

        # Predict the class
        with torch.no_grad():
            outputs = original_model(image)
            
            # Apply softmax to get probabilities
            probabilities = nn.functional.softmax(outputs, dim=1)

            # Get the top 3 predictions and their confidences
            top3_prob, top3_classes = torch.topk(probabilities, 3)

            # Convert top3_classes tensor to list of class names and confidences
            top3_predictions = [(self.classes[class_index], float(prob)) for class_index, prob in zip(top3_classes[0], top3_prob[0])]

            return top3_predictions

        
