import math
import numpy as np
import pandas as pd

results = np.array([[0., 0., 1., 0.],
                    [30., 30., 1., 30.],
                    [45., 45., 1., 45.],
                    [60., 60., 1., 60.],
                    [90., 90., 1., 90.],
                    [60., 120., 2., 120.],
                    [45., 135., 2., 135.],
                    [30., 150., 2., 150.],
                    [0., 180., 2., 180.],
                    [-30., 150., 3., 210.],
                    [-45., 135., 3., 225.],
                    [-60., 120., 3., 240.],
                    [-90., 90., 4., 270.],
                    [-60., 60., 4., 300.],
                    [-45., 45., 4., 315.],
                    [-30., 30., 4., 330.]])
results = pd.DataFrame(results)

r2 = []
for theta in results[3].values:
    theta2 = math.radians(theta)
    xc = math.cos(theta2)
    yc = math.sin(theta2)
    r2.append([theta, theta2, xc, yc])

r2 = pd.DataFrame(r2, columns=["Degrees", "Radians", "x", "y"])
print(r2.round(2))
