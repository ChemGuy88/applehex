"""
Functions module for the package
"""

import math
import xml.etree.ElementTree as ET
# from typing import Union
from datetime import datetime
from xml.etree.ElementTree import ElementTree
from pathlib import Path
# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from sklearn.cluster import KMeans


def parseExportFile(exportFilePath: Path) -> ElementTree:
    """
    Loads the Apple Health export Zip file and returns a parsed XML ElementTree object.
    """
    tree = ET.parse(exportFilePath)  # Parse tree
    return tree


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


def clusterCoordinates(numClusters: int, coordinates: pd.DataFrame) -> KMeans:
    """
    """
    assert numClusters <= 10, "Can only handle up to 10 clusters."  # Because we can only plot up to 10 different colors.
    kmeans = KMeans(n_clusters=numClusters,
                    n_init=1)
    kmeans = kmeans.fit(coordinates.values)

    return kmeans


def time2ordinal(pyTimeObj: datetime.time) -> int:
    """
    Converts a python `datetime.time`-type object into a microseconds-based integer ordinal.
    """
    hours = pyTimeObj.hour
    minutes = pyTimeObj.minute + hours * 60
    seconds = pyTimeObj.second + minutes * 60
    microseconds = pyTimeObj.microsecond + seconds * 10**6
    return microseconds


def plotClusteredCoordinates(model: KMeans, coordinates: pd.DataFrame):
    """
    Visualization of the clustered coordinates.
    """
    figure = plt.figure()
    xc = coordinates["x"].to_list()
    yc = coordinates["y"].to_list()
    plt.scatter(xc, yc, c=model.labels_, marker=".", cmap='Dark2', label=["a", "b"])
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker="x",
                s=169,
                linewidths=3,
                color="b",
                zorder=10)
    plt.title("Title")
    plt.xlabel("Xlabel")
    plt.ylabel("Ylabel")
    plt.legend(["1", "2"])
    return figure


def getAngle(xycoordinates: tuple, test=False) -> float:
    """
    Get the angle from a right triangle defined by x and y-coordinates.
    """
    xycoord = (xycoordinates[0], xycoordinates[1])
    xc = xycoord[0]
    yc = xycoord[1]
    # quadrant = getQuadrant(xc, yc)
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
    if test:
        return (angle0, angle1, quadrant, angle)
    else:
        return angle


def angle2circle(theta: float, radius: int = 1) -> tuple:
    """
    Get the coordinates of the point on a circle with radius r and angle theta in degrees.
    """
    radians = math.radians(theta)
    xc = math.cos(radians) * radius
    yc = math.sin(radians) * radius
    return xc, yc


def projectToCircle(xycoord: tuple) -> tuple:
    """
    Projects an x,y-coordinate to a circle
    """
    theta = getAngle(xycoordinates=xycoord)
    xyproj = angle2circle(theta)
    return xyproj


if __name__ == "__main__":
    # Settings: Interactive Pyplot
    ipythonObj = get_ipython().run_line_magic('matplotlib', "")
