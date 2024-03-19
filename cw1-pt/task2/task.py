import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import os

from network_pt import MyVisionTransformer
from mixup import MixUp  

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_mixup(images, filename='mixup.png'):
    # Assuming `images` is a batch of mixed images from MixUp
    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images, nrow=4)
    img_grid = F.to_pil_image(img_grid)
    img_grid.save(filename)

def train_MixUp(net, trainloader, criterion, optimizer, epochs, alpha, sampling_method, device):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)  # One-hot encode labels
            optimizer.zero_grad()

            # Apply MixUp
            mixup = MixUp(alpha=alpha, sampling_method=sampling_method)
            mixed_inputs, mixed_labels = mixup.mixUp(inputs, labels)
            
            # Forward pass
            outputs = net(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}')

    print('Finished Training')


def main():
    #integrate the training and testing code from tutorial here
    #then run it through the MixUp algorithm and the ViT model and print the results
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testSetBatchSize = 1000
    testloader = torch.utils.data.DataLoader(testset, batch_size=testSetBatchSize, shuffle=False, num_workers=2)

    # Create training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    # Visualise your implementation, by saving to a PNG file “mixup.png”, 
    # a montage of 16 images with randomly augmented images 
    # that are about to be fed into network training.

    # Call ViT network
    net = MyVisionTransformer()  
    net.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train with MixUp for both sampling methods
    for sampling_method in [1, 2]:
        print(f"\nTraining with sampling method {sampling_method}")
        train_MixUp(net, trainloader, criterion, optimizer, epochs=20, alpha=0.4, sampling_method=sampling_method, device=device)
        # Save model
        torch.save(net.state_dict(), f'vit_mixup_sampling_method_{sampling_method}.pt')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    mixup = MixUp(alpha=0.4, sampling_method=1)  # Example use
    mixed_images, _,= mixup.mixUp(images, labels)
    visualize_mixup(mixed_images)

if __name__ == "__main__":
    main()