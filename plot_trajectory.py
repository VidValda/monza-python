import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the simulation data
df = pd.read_csv('simulation_data.csv')

# Extract data
time = df['time'].values
x = df['x'].values
y = df['y'].values

# Create figure with subplots
fig, axes = plt.subplots(1, 1, figsize=(6, 6))

# Plot 1: Trajectory in x-y plane (colored by time)
ax1 = axes
scatter = ax1.scatter(x, y, c=time, cmap='viridis', s=10, alpha=0.6)
ax1.plot(x, y, 'k-', alpha=0.3, linewidth=0.5)  # Connect points with line
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Trajectory in x-y Plane', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.2, 0.2)
ax1.set_ylim(-0.2, 0.2)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')
plt.colorbar(scatter, ax=ax1, label='Time (s)')


# Plot 2: Time evolution of x and y
#ax2 = axes[1]
#ax2.plot(time, x, 'b-', label='x(t)', linewidth=2)
#ax2.plot(time, y, 'r-', label='y(t)', linewidth=2)
#ax2.set_xlabel('Time (s)', fontsize=12)
#ax2.set_ylabel('Position', fontsize=12)
#ax2.set_title('Position vs Time', fontsize=14, fontweight='bold')
#ax2.grid(True, alpha=0.3)
#ax2.legend()

plt.tight_layout()
plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')
print(f"Plot saved as 'trajectory_plot.png'")
print(f"Total simulation time: {time[-1]:.2f} s")
print(f"Number of data points: {len(time)}")
plt.show()

