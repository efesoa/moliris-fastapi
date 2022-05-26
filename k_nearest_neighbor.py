import operator
import similarity_metric


def knn(dataset, obj, k):

    distances = {}
    sort = {}
    length = obj.shape[1]

    # Calculating euclidean distance between each row of dataset and object
    for x in range(len(dataset)):

        dist = similarity_metric.euclidean_distance(obj, dataset.iloc[x], length)
        distances[x] = dist[0]

    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    sorted_d1 = sorted(distances.items())

    neighbors = []

    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]

        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1

    sorted_votes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_votes[0][0], neighbors)
