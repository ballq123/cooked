import numpy as np
import matplotlib.pyplot as plt

# Load image
with np.load("../data/pnp.npz", allow_pickle=True) as data:
    image = data["image"]
    cad = data["cad"][0][0]  # cad is weirdly nested
    cadPoints = cad[0] # points for can model
    cadTriangles = cad[1] # triangles for cad model
    x = data["x"]
    X = data["X"]
cadTriangles -= np.ones_like(cadTriangles) # correct for start at index 1

# write your implementation here
