import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def analyze_dataset():
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Basic dataset statistics
    print("\nDataset Statistics:")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Analyze class distribution
    train_labels = [label for _, label in tqdm(trainset, desc="Analyzing training labels")]
    test_labels = [label for _, label in tqdm(testset, desc="Analyzing test labels")]
    
    train_class_counts = pd.Series(train_labels).value_counts().sort_index()
    test_class_counts = pd.Series(test_labels).value_counts().sort_index()
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, train_class_counts, width, label='Training', alpha=0.7)
    plt.bar(x + width/2, test_class_counts, width, label='Test', alpha=0.7)
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Training and Test Sets')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'class_distribution.png')
    plt.close()
    
    # Analyze image statistics
    train_images = torch.stack([img for img, _ in tqdm(trainset, desc="Analyzing training images")])
    test_images = torch.stack([img for img, _ in tqdm(testset, desc="Analyzing test images")])
    
    # Calculate mean and std for each channel
    train_mean = train_images.mean(dim=(0, 2, 3))
    train_std = train_images.std(dim=(0, 2, 3))
    
    print("\nImage Statistics:")
    print(f"Training set mean: {train_mean.item():.4f}")
    print(f"Training set std: {train_std.item():.4f}")
    
    # Plot sample images from each class
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        # Find first image of this class
        class_idx = np.where(np.array(train_labels) == i)[0][0]
        img, _ = trainset[class_idx]
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'sample_images.png')
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'class': class_names,
        'train_samples': train_class_counts,
        'test_samples': test_class_counts,
        'train_percentage': train_class_counts / len(trainset) * 100,
        'test_percentage': test_class_counts / len(testset) * 100
    })
    stats_df.to_csv(results_dir / 'dataset_statistics.csv', index=False)
    
    return stats_df

if __name__ == "__main__":
    stats = analyze_dataset()
    print("\nDetailed statistics saved to results/dataset_statistics.csv") 