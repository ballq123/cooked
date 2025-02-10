import numpy as np
import matplotlib.pyplot as plt

points = [(10, 10), (20, 20), (30, 30)]
theta = np.linspace(0, 2*np.pi, 360)  
plt.figure(figsize=(8, 6))
for (x, y) in points:
    rho = x * np.cos(theta) + y * np.sin(theta)
    plt.plot(theta, rho, label=f'({x},{y})')

x1, y1 = points[0]
x2, y2 = points[1]
m = (y2 - y1) / (x2 - x1) 
c = y1 - m * x1  

# intercept
t = (3*np.pi) / 4
r = 0

plt.xlabel(r"$\theta$ (radians)")
plt.ylabel(r"$\rho$ (pixels)")
plt.title("Hough Space Sinusoids for Given Points")
plt.scatter([t], [r], color='r', marker='o', label=f'Intercept (θ=3π/4, ρ=0)')
plt.legend()
plt.grid()
plt.savefig("hough_transform_plot.png", dpi=300)
plt.show()
print(f"Intercept in Hough space: rho = {c:.2f}")

