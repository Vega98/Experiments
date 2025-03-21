import numpy as np
import argparse

def inspect_npz(file_path):
    """Inspect the structure and content of an NPZ file."""
    print(f"Inspecting NPZ file: {file_path}")
    print("-" * 50)
    
    # Load the npz file
    data = np.load(file_path)
    
    # List all arrays in the file
    print("Arrays in the file:")
    for array_name in data.files:
        print(f"  - {array_name}")
    print()
    
    # Inspect each array
    for array_name in data.files:
        array = data[array_name]
        print(f"Array: {array_name}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Size (elements): {array.size}")
        print(f"  Memory usage: {array.nbytes / (1024 * 1024):.2f} MB")
        
        # Display sample data
        if array.size > 0:
            if array.ndim == 1:
                print(f"  First 5 elements: {array[:5]}")
            else:
                print(f"  First element shape: {array[0].shape}")
                if array[0].size <= 10:  # Only show small arrays
                    print(f"  First element: {array[0]}")
                else:
                    print(f"  First element (partial): {array[0].flatten()[:5]} ...")
            
            # Count unique classes if this is the 'y' array (labels)
            if array_name == 'y':
                # Flatten in case it's a 2D array like (n, 1)
                flat_array = array.flatten() if array.ndim > 1 else array
                unique_classes = np.unique(flat_array)
                print(f"  Number of unique classes: {len(unique_classes)}")
                print(f"  Class values: {unique_classes}")
                
                # Count samples per class
                class_counts = {}
                for cls in unique_classes:
                    class_counts[cls] = np.sum(flat_array == cls)
                print(f"  Samples per class: {class_counts}")
        print()
    
    # Calculate total size
    total_size_mb = sum(data[array_name].nbytes for array_name in data.files) / (1024 * 1024)
    print(f"Total size: {total_size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ file structure")
    parser.add_argument("file_path", help="Path to the NPZ file to inspect")
    args = parser.parse_args()
    
    inspect_npz(args.file_path)

if __name__ == "__main__":
    main()