import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}

    data = {'reference': [], 'solution': [], 'error': []}
    current_key = None
    current_slice = []

    for line in lines:
        stripped = line.strip()
        
        # Check for section headers
        if stripped == "reference:":
            current_key = 'reference'
            continue
        elif stripped == "solution:":
            current_key = 'solution'
            continue
        elif stripped == "solution error:":
            current_key = 'error'
            continue
        elif stripped.startswith("[FAILURE]"):
            break

        # If we are inside a section
        if current_key:
            if not stripped:
                # Blank line indicates end of a slice
                if current_slice:
                    data[current_key].append(np.array(current_slice))
                    current_slice = []
                continue
            
            # Try to parse row
            row = [float(x) for x in stripped.split()]
            current_slice.append(row)

    # Capture the last slice if file ends without blank line
    if current_slice and current_key:
        data[current_key].append(np.array(current_slice))
    
    return data

def plot_slices(data):
    categories = ['reference', 'solution', 'error', 'ratio']
    # Filter categories that actually have data
    valid_categories = [cat for cat in categories if data.get(cat)]
    
    if not valid_categories:
        print("No valid data found to plot.")
        return

    n_slices = max(len(data[cat]) for cat in valid_categories)
    n_rows = len(valid_categories)
    
    print(f"Plotting {n_rows} rows and up to {n_slices} columns.")

    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(n_slices * 2.5, n_rows * 2.5), squeeze=False)
    
    for i, cat in enumerate(valid_categories):
        slices = data[cat]
        # Label the row
        axes[i, 0].set_ylabel(cat, fontsize=10, fontweight='bold')
        
        for j in range(n_slices):
            ax = axes[i, j]
            if j < len(slices):
                slice_data = slices[j]
                if slice_data.ndim == 2:
                    im = ax.imshow(slice_data, cmap='gray', interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Optional: Add title for column on the first row
                    if i == 0:
                        ax.set_title(f"Slice {j+1}")
                else:
                    ax.text(0.5, 0.5, f"Shape {slice_data.shape}", ha='center', va='center')
                    ax.axis('off')
            else:
                ax.axis('off')

    plt.tight_layout()
    print("Saving plot to slices.png...")
    plt.savefig("slices.png")

if __name__ == "__main__":
    log_file = "full-log.txt"
    print(f"Reading from {log_file}...")
    data = parse_log_file(log_file)
    if any(data.values()):
        plot_slices(data)
    else:
        print("No valid slices found.")
