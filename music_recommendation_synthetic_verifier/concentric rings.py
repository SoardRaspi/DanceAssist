import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Parameters
center = (0, 0)  # Center of the 2D Gaussian
sigma = 1.0      # Standard deviation (same for 1D and 2D)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'brown']  # 7 distinct colors

# Example 1D Gaussian range dictionary
# Each key maps to a tuple: (inner_radius, outer_radius)
gaussian_ring_dict = {
    'ring1': (0.0, 0.5),
    'ring2': (0.5, 1.0),
    'ring3': (1.0, 1.5),
    'ring4': (1.5, 2.0),
    'ring5': (2.0, 2.5),
    'ring6': (2.5, 3.0),
    'ring7': (3.0, 3.5),
}

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Draw rings
for i, (key, (r_inner, r_outer)) in enumerate(gaussian_ring_dict.items()):
    ring = patches.Wedge(center, r_outer, 0, 360, width=r_outer - r_inner, color=colors[i], label=key)
    ax.add_patch(ring)

# Optional: plot 2D Gaussian as background for reference
x = np.linspace(-4*sigma, 4*sigma, 400)
y = np.linspace(-4*sigma, 4*sigma, 400)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2) / (2*sigma**2))

ax.contourf(X, Y, Z, levels=50, cmap='gray', alpha=0.3)

# Final touches
plt.xlim(-4*sigma, 4*sigma)
plt.ylim(-4*sigma, 4*sigma)
plt.legend()
plt.title("Colored Rings Based on 1D Gaussian Ranges")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
