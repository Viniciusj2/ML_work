import h5py
import numpy as np
import argparse

def print_samples(name, obj, num_samples=3):
    """Print sample values from a dataset"""
    if isinstance(obj, h5py.Dataset):
        print(f"\nSamples from '{name}':")
        data = obj[:]
        
        # For 2D arrays, print first few rows and columns
        if len(data.shape) == 2:
            print(f"Shape: {data.shape}")
            print("First few rows:")
            for i in range(min(num_samples, data.shape[0])):
                print(f"Row {i}: {data[i,:min(num_samples, data.shape[1])]}...")
        
        # For 1D arrays, just print first few values
        elif len(data.shape) == 1:
            print(data[:min(num_samples*10, len(data))])
        
        # Print some statistics
        print(f"\nStatistics for '{name}':")
        print(f"  Min: {np.min(data):.4f}")
        print(f"  Max: {np.max(data):.4f}")
        print(f"  Mean: {np.mean(data):.4f}")
        print(f"  Non-zero values: {np.count_nonzero(data)}/{data.size}")

def inspect_h5_file(filename, num_samples=3):
    """Main inspection function"""
    print(f"\nInspecting HDF5 file: {filename}\n")
    print("=" * 60)
    
    try:
        with h5py.File(filename, 'r') as file:
            # Print structure first
            print("\nFile Structure:")
            file.visititems(lambda name, obj: print_h5_structure(name, obj))
            
            # Then print samples from each dataset
            print("\n" + "=" * 60)
            print("\nDataset Samples and Statistics:")
            file.visititems(lambda name, obj: print_samples(name, obj, num_samples))
            
    except Exception as e:
        print(f"\n❌ Error inspecting file: {e}")

def print_h5_structure(name, obj):
    """Print basic structure information"""
    if isinstance(obj, h5py.Group):
        print(f"📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"📊 Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect HDF5 file contents')
    parser.add_argument('filename', type=str, help='Path to HDF5 file')
    parser.add_argument('--samples', type=int, default=3, 
                       help='Number of sample rows to display')
    args = parser.parse_args()
    
    inspect_h5_file(args.filename, args.samples)
