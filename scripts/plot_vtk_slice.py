import numpy as np
import matplotlib.pyplot as plt
import os

def read_custom_vtk(filename):
    """
    Reads the specific ASCII VTK format and automatically detects the grid size (N).
    """
    if not os.path.exists(filename):
        print(f"Error: Could not find {filename}")
        return None, None

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 1. Automatically find N from the header
    N = 0
    for line in lines:
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            N = int(parts[1]) # Extracts the first dimension number
            break
            
    if N == 0:
        print("Error: Could not find DIMENSIONS in VTK header.")
        return None, None

    print(f"Detected Grid Size: {N}x{N}x{N}")

    # 2. Extract the actual floating point data (skipping the first 10 header lines)
    data_1d = np.array([float(line.strip()) for line in lines[10:] if line.strip()])
    
    # 3. Reshape into 3D
    grid_3d = data_1d.reshape((N, N, N))
    return grid_3d, N

# =========================================================
# CONFIGURATION
# =========================================================
filename = "solution_gpu.vtk" # Or "solution_cpu.vtk"

print(f"Loading VTK data from {filename}...")
grid, N = read_custom_vtk(filename)

if grid is not None:
    # Extract the exact middle slice along the Z-axis
    middle_z = N // 2
    slice_2d = grid[middle_z, :, :]

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(8, 6))
    
    # Plot the heat map
    plt.imshow(slice_2d, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
    
    # Add a colorbar and labels
    cbar = plt.colorbar()
    cbar.set_label('Field Potential (u)', rotation=270, labelpad=15)
    
    plt.title(f'Cross-Section of 3D Poisson Solution (Z = {middle_z})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Save the plot as an image and show it
    plt.savefig('poisson_slice_plot.png', dpi=300)
    print("Successfully generated and saved 'poisson_slice_plot.png'")
    plt.show()