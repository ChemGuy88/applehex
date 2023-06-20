"""
"""

import math
import numpy as np
import pandas as pd

li = [(1, 0),
      (np.sqrt(3) / 2, 1 / 2),
      (np.sqrt(2) / 2, np.sqrt(2) / 2),
      (1 / 2, np.sqrt(3) / 2),
      (0, 1),
      (-1 / 2, np.sqrt(3) / 2),
      (-np.sqrt(2) / 2, np.sqrt(2) / 2),
      (-np.sqrt(3) / 2, 1 / 2),
      (-1, 0),
      (-np.sqrt(3) / 2, -1 / 2),
      (-np.sqrt(2) / 2, -np.sqrt(2) / 2),
      (-1 / 2, -np.sqrt(3) / 2),
      (0, -1),
      (1 / 2, -np.sqrt(3) / 2),
      (np.sqrt(2) / 2, -np.sqrt(2) / 2),
      (np.sqrt(3) / 2, -1 / 2)]
results = []
for centroid in li:
    xycoord = (centroid[0], centroid[1])
    xc = xycoord[0]
    yc = xycoord[1]
    if xc < 0:
        if yc < 0:
            quadrant = 3
        elif yc > 0:
            quadrant = 2
    elif xc >= 0:
        if yc < 0:
            quadrant = 4
        elif yc >= 0:
            quadrant = 1
    vector = np.array([xc, yc])
    norm = np.linalg.norm(vector)  # hypoteneuse
    angle0 = math.degrees(math.asin(yc / norm))
    angle1 = math.degrees(math.acos(xc / norm))
    if quadrant in [1, 2]:
        angle = angle1
    elif quadrant in [3, 4]:
        angle = 360 - angle1
    results.append([angle0, angle1, quadrant, angle])
results = pd.DataFrame(results)
print(results)
