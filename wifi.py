from typing import Tuple
import numpy as np

from plot import plot_decision_tree, Decision, TreeNode

LABEL_INDEX: int = 7
NUM_ROOMS: int = 4


def main():
    clean_data = np.loadtxt("./WIFI_db/noisy_dataset.txt", dtype='float')
    for row in clean_data:
        row[LABEL_INDEX] = int(row[LABEL_INDEX])

    print(cross_validate(clean_data))

    # tree = decision_tree_learning(np.transpose(clean_data))
    # # plot_decision_tree(tree, tree.depth)
    # print(tree.depth)
    return 0


def decision_tree_learning(dataT: np.ndarray):
    labels = dataT[LABEL_INDEX]

    # If all labels are the same, no more decisions to be made.
    if np.all(labels[0] == labels):
        return TreeNode(labels[0])

    # Decision tree not perfect,
    (decision, leftData, rightData) = split(dataT)

    left = decision_tree_learning(leftData)
    right = decision_tree_learning(rightData)

    return TreeNode(decision, left, right)


def split(dataT: np.ndarray):
    # loop through every attribute, find split point and highest info gain
    # per attribute
    bestIg = 0
    bestSplit = None

    for attr in range(LABEL_INDEX):
        (ig, split) = attribute_split(dataT, attr)

        if ig > bestIg:
            bestIg = ig
            bestSplit = split

    return bestSplit


# takes in 2 row array (row 0: attribute vals, row 1: labels)
def attribute_split(dataT: np.ndarray, attr: int):
    # (highestIG, midpoint, leftData, rightData)
    # res = (0, 0, [[]], [[]])
    bestIg = 0
    bestSplit = None

    # sort attribute by value (first row)
    sortedDataT = dataT[:, dataT[attr].argsort()]

    # In noisy data the labels are float, but they need to be int
    labels = list(map(int, sortedDataT[LABEL_INDEX]))

    lenData = np.shape(dataT)[1]

    for splitIdx in range(lenData - 1):
        # sliding window of size 2]
        window = sortedDataT[attr][splitIdx: splitIdx + 2]
        midpoint = (window[0] + window[1]) / 2

        # slice sorted 2d array into two parts, seperated by i
        leftData = sortedDataT[:, :splitIdx + 1]
        rightData = sortedDataT[:, splitIdx + 1:]

        ig = infoGain(labels, splitIdx)

        # update result var if new IG is higher
        if ig > bestIg:
            bestIg = ig
            bestSplit = (Decision(attr, midpoint), leftData, rightData)

    return (bestIg, bestSplit)


def infoGain(labels, splitIdx):
    return entropy(labels) - remainder(labels[: splitIdx + 1], labels[splitIdx + 1:])


def remainder(leftLabels: np.array, rightLabels: np.array):
    leftTotal = np.shape(leftLabels)[0]
    rightTotal = np.shape(rightLabels)[0]

    total = leftTotal + rightTotal

    leftRem = (leftTotal / total) * entropy(leftLabels)
    rightRem = (rightTotal / total) * entropy(rightLabels)

    return leftRem + rightRem


def entropy(labels: np.array):

    # as labels array only has values from 1-4, bincount and cut off 0
    occurences = np.bincount(labels)[1:]

    totalExamples = np.shape(labels)[0]

    probabilities = occurences / totalExamples

    entropy = 0

    for p in np.nditer(probabilities):
        if p != 0:
            entropy -= p * np.log2(p)

    return entropy


# data is NOT transpose (each column is an emitter)
# WARNING: mutates data
def cross_validate(data: np.ndarray):
    k = 10

    np.random.shuffle(data)

    splitData = np.split(data, k)

    totalAccuracy = 0

    for i in range(1, k):
        for j in range(1, k):
            # test and validation data cannot be the same
            if i == j:
                continue

            testData = splitData[i]
            validationData = splitData[j]

            tree = decision_tree_learning(np.transpose(validationData))

            totalAccuracy += evaluate(tree, testData)

            break

        break

    return totalAccuracy / k


def evaluate(tree: TreeNode, testData: np.ndarray):
    # Makes a LABEL_INDEX * LABEL_INDEX confusion matrix
    confusionMatrix = np.zeros((NUM_ROOMS, NUM_ROOMS))
    # confusionMatrix[n - 1][m - 1] = number of times the phone was in room n,
    # and we predicted it was in room m

    for [*strengths, actualRoom] in testData:
        predictedRoom = tree.get_room(strengths)

        confusionMatrix[int(actualRoom) - 1][predictedRoom - 1] += 1

    # totals
    totals = np.zeros((NUM_ROOMS, 4))

    for roomIdx in range(NUM_ROOMS):
        truePositive = confusionMatrix[roomIdx][roomIdx]

        # True positive.
        totals[roomIdx][0] += truePositive
        # True negative.
        totals[roomIdx][1] += np.sum(np.diagonal(confusionMatrix)
                                     ) - truePositive
        # False positive.
        totals[roomIdx][2] += np.sum(confusionMatrix[roomIdx]) - truePositive
        # False negative.
        totals[roomIdx][3] += np.sum(np.transpose(confusionMatrix)
                                     [roomIdx]) - truePositive

    statistics = np.zeros((NUM_ROOMS, 4))
    # statistics[n - 1][0] = precision for room n
    # statistics[n - 1][1] = recall for room n
    # statistics[n - 1][2] = accuracy for room n
    # statistics[n - 1][3] = f1 score for room n

    for roomIdx in range(NUM_ROOMS):
        statistics[roomIdx][0] = totals[roomIdx][0] / \
            (totals[roomIdx][0] + totals[roomIdx][2])
        statistics[roomIdx][1] = totals[roomIdx][0] / \
            (totals[roomIdx][0] + totals[roomIdx][3])
        statistics[roomIdx][2] = (totals[roomIdx][0] +
                                  totals[roomIdx][1]) / np.sum(totals[roomIdx])
        statistics[roomIdx][3] = 2 * (statistics[roomIdx][0] *
                                      statistics[roomIdx][1]) / (statistics[roomIdx][0] + statistics[roomIdx][1])

    print("CONFUSION MATRIX")
    print(confusionMatrix)
    print("TOTALS")
    print(totals)
    print("STATISTICS")
    print(statistics)

    return 0


if __name__ == "__main__":
    main()
