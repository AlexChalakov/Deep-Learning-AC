import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import torchvision.transforms.functional as F

from network_pt import MyVisionTransformer
from mixup import MixUp  

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_MixUp(net, trainloader, valloader, holdoloader, criterion, optimizer, epochs, alpha, sampling_method, device):
    net.train()

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)  # One-hot encode labels
            optimizer.zero_grad()

            # Apply MixUp
            mixup = MixUp(alpha=alpha, sampling_method=sampling_method)
            mixed_inputs, mixed_labels = mixup.mixUp(images, labels)
            
            # Forward pass
            outputs = net(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print every batch for testing
            print(f'Batch [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(mixed_labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_number = epoch + 1
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        epoch_speed = len(trainloader) / epoch_time
        print(f'Epoch [{epoch_number}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}, Speed: {epoch_speed:.2f}%')

        # Validate the network
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                val_outputs = net(images)
                val_loss = criterion(val_outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
                val_running_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        # Print accuracy and loss on the validation set
        val_mean_loss = val_running_loss / len(valloader)
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Loss: {val_mean_loss:.2f}, Validation Accuracy: {val_accuracy:.2f} %')

        # Test the network w/ the holdout test set
        test_correct = 0
        test_total = 0
        test_running_loss = 0.0
        with torch.no_grad():
            for images, labels in holdoloader:
                images, labels = images.to(device), labels.to(device)
                test_outputs = net(images)
                test_loss = criterion(test_outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
                test_running_loss += test_loss.item()

                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (test_predicted == labels).sum().item()
        
        # Print accuracy and loss on the holdout test set
        test_mean_loss = test_running_loss / len(holdoloader)
        test_accuracy = 100 * test_correct / test_total
        print(f'Test Loss: {test_mean_loss:.2f}, Test Accuracy: {test_accuracy:.2f} %')

    print('Finished Training, Validating and Testing!')
    print('Task 3 complete')

def main():
    # Integrate the training and testing code from tutorial here
    # then run it through the MixUp algorithm and the ViT model and print the results
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Random split the dataset into development set (80%) and holdout test set (20%).
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    length = len(dataset)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    development_size = int(0.8 * length)
    holdout_size = length - development_size
    development_set, holdout_testset = random_split(dataset, [development_size, holdout_size])

    # Random split the development_set into train (90%) and validation sets (10%).
    train_size = int(0.9 * development_size)
    validation_size = development_size - train_size
    trainset, validation_set = random_split(development_set, [train_size, validation_size])

    trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
    valloader = DataLoader(validation_set, batch_size=50, shuffle=True, num_workers=2)
    holdoloader = DataLoader(holdout_testset, batch_size=36, shuffle=False, num_workers=2)

    # Call ViT network
    net = MyVisionTransformer()  
    net.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train with MixUp for both sampling methods
    for sampling_method in [1, 2]:
        print(f"\nTraining with sampling method {sampling_method}")
        # Train with MixUp - EPOCH IS 1 FOR TESTING
        train_MixUp(net, trainloader, valloader, holdoloader, criterion, optimizer, epochs=1, alpha=0.4, sampling_method=sampling_method, device=device)
        # Save the model
        torch.save(net.state_dict(), f'vit_mixup_sampling_method_{sampling_method}.pt')

if __name__ == "__main__":
    main()