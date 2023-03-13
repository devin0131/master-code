import math


def max_distance(points):
    max_dist = 0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0]-points[j][0])**2 +
                             (points[i][1]-points[j][1])**2 +
                             (points[i][2]-points[j][2])**2)
            if dist > max_dist:
                max_dist = dist
    return max_dist
