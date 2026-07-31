import matplotlib.pyplot as plt
import subprocess
import re
import sys
import os

# =========================================================
# CONFIGURATION
# =========================================================
# The grid sizes (N) you want to test. (N=64 means a 64x64x64 grid)
grid_sizes = [16, 32, 48, 64] 

# Paths to your executables (adjust if your script is in a different folder)
cpu_exec = "./build/main_cpu"
gpu_exec = "./build/main_cuda"

# Check if executables exist
if not os.path.exists(cpu_exec) or not os.path.exists(gpu_exec):
    print(f"Error: Cannot find {cpu_exec} or {gpu_exec}.")
    print("Make sure you compile the code first and run this script from the project root.")
    sys.exit(1)

cpu_times = []
gpu_times = []

# =========================================================
# AUTOMATED DATA COLLECTION
# =========================================================
print("Starting Automated Benchmark...")

for N in grid_sizes:
    print(f"\n--- Testing Grid Size: {N}x{N}x{N} ---")
    
    # 1. Run CPU Solver
    print("Running CPU...")
    # subprocess.run executes the terminal command: ./build/main_cpu N
    cpu_result = subprocess.run([cpu_exec, str(N)], capture_output=True, text=True)
    
    # Extract the time using Regex (Looks for "Execution Time: [number] seconds")
    cpu_match = re.search(r'Time:\s*([0-9.]+)\s*seconds', cpu_result.stdout)
    if cpu_match:
        time = float(cpu_match.group(1))
        cpu_times.append(time)
        print(f"CPU Time: {time} s")
    else:
        print("Error reading CPU time. Did the solver fail?")
        cpu_times.append(0)

    # 2. Run GPU Solver
    print("Running GPU...")
    gpu_result = subprocess.run([gpu_exec, str(N)], capture_output=True, text=True)
    
    gpu_match = re.search(r'Time:\s*([0-9.]+)\s*seconds', gpu_result.stdout)
    if gpu_match:
        time = float(gpu_match.group(1))
        gpu_times.append(time)
        print(f"GPU Time: {time} s")
    else:
        print("Error reading GPU time.")
        gpu_times.append(0)

# Calculate Speedup
speedup = [c / g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]

# =========================================================
# PLOTTING
# =========================================================
print("\nGenerating Plot...")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-Axis: Execution Time (Log Scale)
color = 'tab:red'
ax1.set_xlabel('Grid Size (N) for NxNxN Grid')
ax1.set_ylabel('Execution Time (seconds)', color=color)
ax1.plot(grid_sizes, cpu_times, marker='o', label='Intel i5 (CPU)', color='red', linestyle='--')
ax1.plot(grid_sizes, gpu_times, marker='s', label='NVIDIA MX230 (GPU)', color='darkred')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.legend(loc='upper left')

# Secondary Y-Axis: Speedup Multiplier
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('GPU Speedup (CPU Time / GPU Time)', color=color)  
ax2.plot(grid_sizes, speedup, marker='^', label='Speedup (x)', color='blue', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, max(speedup) * 1.2) 
ax2.legend(loc='upper center')

# Title and Layout
plt.title('3D Poisson CGLS: CPU vs GPU Performance Scaling')
plt.xticks(grid_sizes)
fig.tight_layout()

# Save and Show
plt.savefig('performance_scaling.png', dpi=300)
print("Done! Plot successfully saved as 'performance_scaling.png'")
plt.show()