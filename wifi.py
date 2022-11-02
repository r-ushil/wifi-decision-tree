from typing import Tuple
import numpy as np

from plot import plot_decision_tree, Decision, TreeNode

LABEL_INDEX: int = 7


def main():
    clean_data = np.loadtxt("./WIFI_db/noisy_dataset.txt", dtype='float')
    (tree, depth) = decision_tree_learning(clean_data, 0)
    plot_decision_tree(tree, depth)
    print(depth)
    return 0


def decision_tree_learning(data: np.ndarray, depth: int):

    label = np.transpose(data)[LABEL_INDEX]

    # if all samples in data have same label (col 7)
    if np.all(label[0] == label):
        return (TreeNode(label[0], depth), depth)

    (decision, leftData, rightData) = split(data)

    node = TreeNode(decision, depth)

    leftBranch, leftDepth = decision_tree_learning(leftData, depth+1)
    rightBranch, rightDepth = decision_tree_learning(rightData, depth+1)

    node.left = leftBranch
    node.right = rightBranch

    return (node, max(leftDepth, rightDepth))


def split(data: np.ndarray):

    # loop through every attribute, find split point and highest info gain
    # per attribute

    bestIG = 0
    splitVal = 0
    bestAttr = -1
    bestLeftData = bestRightData = None

    dataT = np.transpose(data)

    for attr in range(LABEL_INDEX):

        (ig, decision, leftData, rightData) = attribute_split(dataT, attr)

        if ig > bestIG:
            bestIG = ig
            bestDecision = decision
            bestLeftData = leftData
            bestRightData = rightData

    return (bestDecision, np.transpose(bestLeftData), np.transpose(bestRightData))


# takes in 2 row array (row 0: attribute vals, row 1: labels)
def attribute_split(dataT: np.ndarray, attr: int):

    # (highestIG, midpoint, leftData, rightData)
    # res = (0, 0, [[]], [[]])
    res = None
    highestIG = 0

    # sort attribute by value (first row)
    sortedDataT = dataT[:, dataT[attr].argsort()]

    noOfExamples = np.shape(dataT)[1]

    for i in range(noOfExamples - 1):

        # sliding window of size 2]
        valPair = sortedDataT[attr][i: i + 2]

        midpoint = (valPair[0] + valPair[1]) / 2

        # slice sorted 2d array into two parts, seperated by i
        leftData = sortedDataT[:, :i+1]
        rightData = sortedDataT[:, i+1:]

        ig = infoGain(
            sortedDataT[LABEL_INDEX], leftData[LABEL_INDEX], rightData[LABEL_INDEX])

        # update result var if new IG is higher
        if ig > highestIG:
            highestIG = ig
            res = (highestIG, midpoint, leftData, rightData)

    return res


def infoGain(dataLabels, leftLabels, rightLabels):
    return entropy(dataLabels) - remainder(leftLabels, rightLabels)


def remainder(leftLabels: np.array, rightLabels: np.array):

    leftTotal = np.shape(leftLabels)[0]
    rightTotal = np.shape(rightLabels)[0]
    combinedTotal = leftTotal + rightTotal

    leftRem = (leftTotal / combinedTotal) * entropy(leftLabels)
    rightRem = (rightTotal / combinedTotal) * entropy(rightLabels)

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
