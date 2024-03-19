import torch
import numpy as np
import torchvision
#import matplotlib.pyplot as plt

class MixUp:
    def __init__(self, alpha, sampling_method):
        self.alpha = alpha
        self.sampling_method = sampling_method

    def lambda_options(self):
        """
        Returns the lambda options for the mixup method

        Output:
        - lambda: lambda options
        """
        if self.sampling_method == 1:
            # Sampling λ from a beta distribution as described in the paper
            return np.random.beta(self.alpha, self.alpha)
        elif self.sampling_method == 2:
            # Sampling λ uniformly from a predefined range 0 to 0.5
            return np.random.uniform(0, 0.5)

    def mixUp(self, x, y):
        """
        Mixes the input data with the labels as described in the https://arxiv.org/pdf/1710.09412.pdf paper

        Inputs:
        - x: images
        - y: labels

        Output:
        - mixed_images: mixed images
        - mixed_labels: mixed labels
        """
        batch_size = x.size(0)
        lambdaInput = self.lambda_options()

        # Permuting the batch to mix with
        perm = torch.randperm(batch_size).to(x.device)
        mixed_images = lambdaInput * x + (1 - lambdaInput) * x[perm]
        mixed_labels = lambdaInput * y + (1 - lambdaInput) * y[perm]

        return mixed_images, mixed_labels

    #def visualize_mixUp(self, images, save_path='mixup.png'):
        # Assuming images is a batch of 16 images
        _, mixed_labels = self.mixUp(images, images)  # Dummy labels, not used here
        grid_img = torchvision.utils.make_grid(mixed_labels, nrow=4)
        npimg = grid_img.numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(save_path)