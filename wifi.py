from typing import Tuple
import numpy as np

from plot import plot_decision_tree, Decision, TreeNode

LABEL_INDEX: int = 7


def main():
    clean_data = np.loadtxt("./WIFI_db/clean_dataset.txt", dtype='int')
    tree = decision_tree_learning(np.transpose(clean_data))
    # plot_decision_tree(tree, tree.depth)
    print(tree.depth)
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

    lenData = np.shape(dataT)[1]

    for splitIdx in range(lenData - 1):
        # sliding window of size 2]
        window = sortedDataT[attr][splitIdx: splitIdx + 2]
        midpoint = (window[0] + window[1]) / 2

        # slice sorted 2d array into two parts, seperated by i
        leftData = sortedDataT[:, :splitIdx + 1]
        rightData = sortedDataT[:, splitIdx + 1:]

        ig = infoGain(sortedDataT[LABEL_INDEX], splitIdx)

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


if __name__ == "__main__":
    main()
