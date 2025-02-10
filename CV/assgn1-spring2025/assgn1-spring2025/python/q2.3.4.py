import numpy as np
from matplotlib import pyplot as plt


points = [(10, 10), (20, 20), (30, 30)]
theta = np.linspace(0, np.pi, 1000)
plt.figure(figsize=(8, 6))

rho_curves = []
for (x, y) in points:
    rho = x * np.cos(theta) + y * np.sin(theta)
    rho_curves.append(rho)
    plt.plot(theta, rho, label=f"({x},{y})")

plt.xlabel(r'$\theta$ (radians)')
plt.ylabel(r'$\rho$ (pixels)')
plt.title("Hough Space Sinusoids for Given Points")
plt.legend()
plt.grid()

rho_curves = np.array(rho_curves)
differences = np.abs(rho_curves[0] - rho_curves[1]) + np.abs(rho_curves[0] - rho_curves[2])
min_index = np.argmin(differences)

theta_intersect = theta[min_index]
rho_intersect = rho_curves[0][min_index]

print(theta_intersect, rho_intersect)
plt.scatter(theta_intersect, rho_intersect, color='red', zorder=3, label=f"Intersection Point: ({2.355}, {0.011})")
plt.legend()
plt.savefig("../results/houghSpacePlot.png", dpi=300, bbox_inches='tight')
plt.show()

