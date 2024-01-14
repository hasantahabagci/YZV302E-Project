import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import cv2
import os
from PIL import Image


class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)


    def forward(self, x):
        return self.model(x)



class VGG16Model():
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
                    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG16
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalization for VGG16
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

        self.model = VGG16(self.num_classes).to(self.device)

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

            # Validation loop (if you have a validation set)
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
                    torch.save(self.model.state_dict(), 'models/vgg16_model.pth')
                    print(f'Saved new best model with accuracy: {best_val_accuracy:.2f}%')

        return self.model


    def predict(self, image):
        vgg16_model = VGG16(self.num_classes, pretrained=False).to(self.device)
        vgg16_model.load_state_dict(torch.load('models/vgg16_model.pth', map_location=self.device))
        vgg16_model.eval()
        image = Image.fromarray(image)
        transform = self.transform()

        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = vgg16_model(image)
            
            # Apply softmax to get probabilities
            probabilities = nn.functional.softmax(outputs, dim=1)

            # Get the top 3 predictions and their confidences
            top3_prob, top3_classes = torch.topk(probabilities, 3)

            # Convert top3_classes tensor to list of class names and confidences
            top3_predictions = [(self.classes[class_index], float(prob)) for class_index, prob in zip(top3_classes[0], top3_prob[0])]

            return top3_predictions
        
