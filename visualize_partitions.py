import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import ListedColormap

def load_partitions(base_dir, partitioner_name):
    """Load all partitions from a directory for a specific partitioner."""
    partitions_dir = os.path.join(base_dir, partitioner_name)
    partition_files = sorted([f for f in os.listdir(partitions_dir) if f.endswith('.npz')])
    
    partitions_data = []
    for file in partition_files:
        file_path = os.path.join(partitions_dir, file)
        data = np.load(file_path)
        y = data['y']
        
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.flatten()
            
        partitions_data.append(y)
    
    return partitions_data

def analyze_class_distribution(partitions_data, num_classes=10):
    """Calculate the class distribution for each partition."""
    num_partitions = len(partitions_data)
    distribution = np.zeros((num_partitions, num_classes))
    
    for i, partition in enumerate(partitions_data):
        # Count occurrences of each class
        for class_idx in range(num_classes):
            count = np.sum(partition == class_idx)
            distribution[i, class_idx] = count
            
        # Normalize to percentage
        if np.sum(distribution[i]) > 0:
            distribution[i] = distribution[i] / np.sum(distribution[i]) * 100
            
    return distribution

def plot_class_distribution(distributions_dict, num_classes=10, class_names=None):
    """Plot class distribution for multiple partitioning schemes."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Set up the plot
    num_schemes = len(distributions_dict)
    fig, axes = plt.subplots(1, num_schemes, figsize=(5*num_schemes, 10), sharey=True)
    
    # Generate a red-to-green colormap
    colors = plt.colormaps['RdYlGn']
    
    # If only one scheme, axes won't be an array
    if num_schemes == 1:
        axes = [axes]
    
    # Plot each partitioning scheme
    for i, (scheme_name, distribution) in enumerate(distributions_dict.items()):
        ax = axes[i]
        num_partitions = distribution.shape[0]
        
        # Create stacked horizontal bar chart
        left = np.zeros(num_partitions)
        for class_idx in range(num_classes):
            ax.barh(range(num_partitions), distribution[:, class_idx], 
                   left=left, color=colors(class_idx/num_classes), height=0.8)
            left += distribution[:, class_idx]
        
        # Set labels and title
        ax.set_title(scheme_name)
        ax.set_xlabel('Class distribution')
        if i == 0:
            ax.set_ylabel('Partition ID')
        ax.set_yticks(range(num_partitions))
        ax.set_yticklabels(range(num_partitions))
        ax.set_xlim(0, 100)
    
    plt.suptitle("Class Distribution in our CIFAR10 Partitions", fontsize=16)
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize partitioning schemes distribution")
    parser.add_argument("--base_dir", required=True, help="Base directory containing partitioning folders")
    parser.add_argument("--output", default="partition_comparison.png", help="Output image file")
    parser.add_argument("--dataset", default="CIFAR10", help="Dataset name")
    args = parser.parse_args()
    
    # CIFAR10 class names
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                      "dog", "frog", "horse", "ship", "truck"]
    
    # Find all partitioning schemes (folders in the base directory)
    partitioning_schemes = [d for d in os.listdir(args.base_dir) 
                           if os.path.isdir(os.path.join(args.base_dir, d))]
    
    # Load and analyze partitions for each scheme
    distributions = {}
    for scheme in partitioning_schemes:
        try:
            partitions_data = load_partitions(args.base_dir, scheme)
            distribution = analyze_class_distribution(partitions_data, 
                                                     num_classes=len(cifar10_classes))
            distributions[scheme] = distribution
        except Exception as e:
            print(f"Error loading {scheme}: {e}")
    
    # Plot the distributions
    fig = plot_class_distribution(distributions, 
                                 num_classes=len(cifar10_classes), 
                                 class_names=cifar10_classes)
    
    # Save the figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {args.output}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()