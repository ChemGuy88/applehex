"""
Extract data from Apple Health app.
"""

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.cm import get_cmap
# Local packages
from functions import parseExportFile, getRecordTypes, getRecordsByAttributeValue, tabulateRecords, timestampsToCoordinates, clusterCoordinates, projectToCircle

# Arguments
DATA_FILE_PATH = Path("data/input/apple_health_export/export.xml")

# Load exported Apple Health data
tree = parseExportFile(DATA_FILE_PATH)

# Get record types
recordTypes = getRecordTypes(tree)

# Get systolic and diastolic blood pressure records
recordsSBP = getRecordsByAttributeValue(tree=tree,
                                        attribute="type",
                                        value="HKQuantityTypeIdentifierBloodPressureSystolic")

recordsDBP = getRecordsByAttributeValue(tree=tree,
                                        attribute="type",
                                        value="HKQuantityTypeIdentifierBloodPressureDiastolic")

# Tabulate blood pressure
dfSBP = tabulateRecords(records=recordsSBP)
dfDBP = tabulateRecords(records=recordsDBP)

# Machine-learning pre-processing: Convert time to x and y-coordinates
times = pd.to_datetime(dfSBP["startDate"])
coordinates = timestampsToCoordinates(times=times)

# Cluster records by time
kmeans = clusterCoordinates(numClusters=2,
                            coordinates=coordinates)

# Plot model
plt.close('all')
figure = plt.figure()
xc = coordinates["x"]
yc = coordinates["y"]
uniqueLabels = sorted(list(set(kmeans.labels_)))
centroids = kmeans.cluster_centers_
projectedCentroids = np.array([projectToCircle(centroid) for centroid in centroids])
labelsDict = {label: (centroid[0], centroid[1]) for centroid, label in zip(centroids, uniqueLabels)}
caseLabels = kmeans.labels_
colors = get_cmap("tab10").colors
for it, (group, centroid) in enumerate(labelsDict.items()):
    mask = caseLabels == group
    groupName = group + 1
    centroidRounded = (np.round(centroid[0], 2), (np.round(centroid[1], 2)))
    plt.scatter(xc[mask], yc[mask], c=colors[it], marker=".", label=f"Group {groupName}: {centroidRounded}")
plt.scatter(centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="b",
            zorder=10,
            label="centroids")
plt.scatter(projectedCentroids[:, 0],
            projectedCentroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="r",
            zorder=10,
            label="Record Mean")  # "Record Mean" is actually the projected centroid.
# Add line from center to circle, for troubleshooting.
origin = np.array([[0, 0] for _ in projectedCentroids])
plt.quiver(*origin,
           projectedCentroids[:, 0],
           projectedCentroids[:, 1],
           color="k",
           angles='xy',
           scale_units='xy',
           scale=1,
           width=0.0035,
           label="Projected Centroid Vector")
plt.plot()
plt.title("24-hour distribution of records")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.legend(loc="best")
# figure = plotClusteredCoordinates(model=kmeans,
#                                   coordinates=coordinates)

# TODO Convert centroids to time
# - Project centroids to unit circle
# - Convert x, y coordinate to time

# TODO Calculate radian offset
