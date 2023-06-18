"""
Extract data from Apple Health app.
"""

import math
import xml.etree.ElementTree as ET
# from typing import Union
from datetime import datetime
from xml.etree.ElementTree import ElementTree
from pathlib import Path
# Third-party packages
import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython
from sklearn.cluster import KMeans

DATA_FILE_PATH = Path("data/input/apple_health_export/export.xml")

# Settings: Interactive Pyplot
get_ipython().run_line_magic('matplotlib', "")


def getRecordTypes(tree: ElementTree) -> set:
    """
    """
    types = set()
    for record in tree.getiterator('Record'):
        attributes = record.attrib
        if 'type' in attributes.keys():
            attributeValue = attributes['type']
            types.add(attributeValue)
    return types


def getRecordsByAttributeValue(tree: ElementTree, attribute: str, value: str) -> list:
    """
    """
    path = f""".//*[@{attribute}='{value}']"""
    records = tree.findall(path=path)
    return records


def tabulateRecords(records: "list[ElementTree]") -> pd.DataFrame:
    """
    Converts a list of records to a Pandas dataframe
    """
    df = pd.DataFrame([record.attrib for record in records])
    return df


def projectTimeToCircle(time: datetime) -> "tuple[float, float]":
    """
    Maps a 24-hour `datetime` time to its corresponding coordinate on the unit circle.
    | time      |   x   |   y   |
    |(hh:mm:ss) |       |       |
    | --------- | ----- | ----- |
    | 00:00:00  |   0   |   1   |
    | 06:00:00  |   1   |   0   |
    | 12:00:00  |   0   |  -1   |
    | 18:00:00  |  -1   |   0   |
    | 24:00:00  |   0   |   1   |
    """
    hours = time.hour
    minutes = time.minute
    seconds = time.second
    timeSeconds = 60 * (hours * 60 + minutes) + seconds
    RADIANS_PER_DAY = 2 * math.pi / 86400
    RADIANS_OFFSET = 0
    radians = timeSeconds * RADIANS_PER_DAY + RADIANS_OFFSET
    xcoord = math.cos(radians)
    ycoord = math.sin(radians)
    return (xcoord, ycoord)


def timestampsToCoordinates(times: pd.Series) -> pd.DataFrame:
    """
    Converts a pandas series of timestamps into a pandas dataframe of x and y coordinates
    """
    li = []
    for _, time in times.iteritems():
        li.append(projectTimeToCircle(time))
    coordinates = pd.DataFrame(li, columns=["x", "y"])
    return coordinates


# Parse tree
tree = ET.parse(DATA_FILE_PATH)
root = tree.getroot()

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
NUMBER_OF_MEASUREMENTS_PER_DAY = 2
kmeans = KMeans(n_clusters=NUMBER_OF_MEASUREMENTS_PER_DAY,
                n_init=1)
kmeans = kmeans.fit(coordinates.values)

# Plot model
plt.close()
fig = plt.figure()
xc = coordinates["x"].to_list()
yc = coordinates["y"].to_list()
plt.scatter(xc, yc, c=kmeans.labels_, marker=".", cmap='Dark2')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="b",
            zorder=10)

# TODO Convert centroids to time
# - Project centroids to unit circle
# - - Get angle
# - - From angle compute x and y coordinates
# - Convert x, y coordinate to time

# TODO Calculate radian offset
