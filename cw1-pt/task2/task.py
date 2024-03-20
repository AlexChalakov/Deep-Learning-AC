import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as F

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

def train_MixUp(net, trainloader, testloader, criterion, optimizer, epochs, alpha, sampling_method, device):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
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

            # Print every batch
            print(f'Batch [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}')

        # Test the network
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f} %')

    print('Finished Training')


def main():
    # Integrate the training and testing code from tutorial here
    # then run it through the MixUp algorithm and the ViT model and print the results
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=36, shuffle=False, num_workers=2)

    # Create training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
    batches = len(trainloader)

    # Visualise your implementation, by saving to a PNG file “mixup.png”, 
    # a montage of 16 images with randomly augmented images 
    # that are about to be fed into network training.
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    mixup = MixUp(alpha=0.4, sampling_method=1)
    mixed_images, _,= mixup.mixUp(images, labels)
    visualize_mixup(mixed_images)

    # Call ViT network
    net = MyVisionTransformer()  
    net.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train with MixUp for both sampling methods
    for sampling_method in [1, 2]:
        print(f"\nTraining with sampling method {sampling_method}")
        print(batches)
        # Train with MixUp - EPOCH IS 1 FOR TESTING
        train_MixUp(net, trainloader, testloader, criterion, optimizer, epochs=1, alpha=0.4, sampling_method=sampling_method, device=device)
        # Save model
        torch.save(net.state_dict(), f'vit_mixup_sampling_method_{sampling_method}.pt')

    # Visualising results with printed messages
    # indicating the ground-truth and the predicted classes for each.
    dataiter(testloader)
    images, labels = next(dataiter)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("result.png")

    # Print ground truth labels
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Print predicted labels
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

if __name__ == "__main__":
    main()