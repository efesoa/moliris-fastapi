import numpy as np
from numpy import dot


def euclidean_distance(a, b, length: int) -> float:
    distance = 0
    for x in range(length):
        distance += np.square(a[x] - b[x])

    return round(np.sqrt(distance), 2)


def cosine_similarity(a, b) -> float:
    return round(dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5)), 2)


def manhattan(a, b):
    return sum(abs(s1 - s2) for s1, s2 in zip(a, b))
