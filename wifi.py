import numpy as np
import sys
from tree import TreeBranch, TreeLeaf, Decision

from plot import plot_dtree

LABEL_INDEX: int = 7
NUM_ROOMS: int = 4


def decision_tree_learning(dataT: np.ndarray):
    labels = dataT[LABEL_INDEX]

    # If all labels are the same, no more decisions to be made.
    # Return a leaf node with that label and no of occurences of that label.
    if np.all(labels[0] == labels):
        room = int(labels[0])
        return TreeLeaf(room, {room: np.shape(labels)[0]})

    # Decision tree not perfect,
    (decision, leftData, rightData) = split(dataT)

    left = decision_tree_learning(leftData)
    right = decision_tree_learning(rightData)

    return TreeBranch(decision, left, right)


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
    labels = np.int_(sortedDataT[LABEL_INDEX])

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
# WARNING: mutates data when shuffling
def cross_validate(data: np.ndarray):
    k = 10

    np.random.seed(42)  # for reproducibility
    np.random.shuffle(data)

    splitData = np.split(data, k)

    totalConfusionMatrix = np.zeros((NUM_ROOMS, NUM_ROOMS))
    totalStatistics = np.zeros((NUM_ROOMS, 4))
    totalAccuracy = 0

    for i in range(0, k):

        testData = splitData[i]
        trainingFolds = splitData[:]
        del trainingFolds[i]
        trainingData = np.concatenate(trainingFolds)

        tree = decision_tree_learning(np.transpose(trainingData))
        confusionMatrix, statistics, accuracy = evaluate(tree, testData)
        totalConfusionMatrix += confusionMatrix
        totalStatistics += statistics
        totalAccuracy += accuracy

        overfit_tree_output_file = f'../out/overfit/tree{i}.png'
        plot_dtree(tree, overfit_tree_output_file)

    # divide by number of test sets
    totalConfusionMatrix /= k
    totalStatistics /= k
    totalAccuracy /= k

    print(totalConfusionMatrix)
    print(totalStatistics)
    print(totalAccuracy)

    return (totalConfusionMatrix, totalStatistics)


def confusion_matrix(tree, testData: np.ndarray):
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
        totals[roomIdx][3] += np.sum(confusionMatrix[:,
                                     roomIdx]) - truePositive
    return confusionMatrix, totals


def evaluate(tree, testData: np.ndarray):
    confusionMatrix, totals = confusion_matrix(tree, testData)

    statistics = np.zeros((NUM_ROOMS, 4))
    # statistics[n - 1][0] = precision for room n
    # statistics[n - 1][1] = recall for room n
    # statistics[n - 1][2] = accuracy for room n
    # statistics[n - 1][3] = f1 score for room n

    for roomIdx in range(NUM_ROOMS):

        tp = totals[roomIdx][0]
        tn = totals[roomIdx][1]
        fp = totals[roomIdx][2]
        fn = totals[roomIdx][3]

        # Precision = tp / (tp + fp)
        statistics[roomIdx][0] = tp / (tp + fp)
        # Recall = tp / (tp + fn)
        statistics[roomIdx][1] = tp / (tp + fn)
        # Accuracy = (tp + tn) / (tp + tn + fp + fn)
        statistics[roomIdx][2] = (tp + tn) / (tp + tn + fp + fn)
        # F1 score = 2 * (precision * recall) / (precision + recall)
        statistics[roomIdx][3] = 2 * (statistics[roomIdx][0] * statistics[roomIdx][1]) / (
            statistics[roomIdx][0] + statistics[roomIdx][1])

    avgAccuracy = np.sum(statistics[:, 2]) / NUM_ROOMS
    return (confusionMatrix, statistics, avgAccuracy)


def leaf_accuracy(leaf: TreeLeaf, validationData: np.ndarray):
    if validationData.size == 0:
        return 1

    # node is a leaf - figure out how accurate it is

    # calculate accuracy by figuring out how much
    # of the validation data has this label
    labels = validationData[:, -1]
    correct = np.sum(labels == leaf.room)
    total = np.shape(labels)[0]

    return correct / total


# PRE: branch.left and branch.right are leaves
def branch_accuracy(branch: TreeBranch, validationData: np.ndarray):
    if validationData.size == 0:
        return 1

    # Node is a tree - figure out how accurate it is
    leftValidationData, rightValidationData = split_validation(
        branch.decision, validationData)
    leftAccuracy = leaf_accuracy(branch.left, leftValidationData)
    rightAccuracy = leaf_accuracy(branch.right, rightValidationData)

    return (leftAccuracy + rightAccuracy) / 2

# prunes decision tree against validationData
# NOTE: Shuffles validationData
# NOTE: mangles tree
# returns pruned tree


def prune(tree, validationData: np.ndarray):
    if tree.is_leaf():
        return tree

    # Prune children
    leftValidationData, rightValidationData = split_validation(
        tree.decision, validationData)
    tree.left = prune(tree.left, leftValidationData)
    tree.right = prune(tree.right, rightValidationData)

    # Try prune this node
    if tree.left.is_leaf() and tree.right.is_leaf():
        leafReplacingNode = tree.left.merge_leaves(tree.right)

        # accuracy if we replace node with leaf
        leafAccuracy = leaf_accuracy(leafReplacingNode, validationData)

        # accuracy if we don't
        treeAccuracy = branch_accuracy(tree, validationData)

        if leafAccuracy > treeAccuracy:
            # In place replaces tree with the leaf
            return leafReplacingNode

    return tree


# TODO: split it in place, then return slices for better performance
# NOTE: (Can) shuffle validationData
def split_validation(decision: Decision, validationData: np.ndarray):
    col = decision.emitter
    val = decision.value
    data = validationData

    leftValidationData = data[data[:, col] < val]
    rightValidationData = data[data[:, col] >= val]

    return leftValidationData, rightValidationData


# data is NOT transpose (each column is an emitter)
# WARNING: mutates data when shuffling
def cross_validate_pruning(data: np.ndarray):
    k = 10

    np.random.seed(42)  # for reproducibility
    np.random.shuffle(data)

    splitData = np.split(data, k)

    totalConfusionMatrix = np.zeros((NUM_ROOMS, NUM_ROOMS))
    totalStatistics = np.zeros((NUM_ROOMS, 4))
    totalAccuracy = 0
    count = 0

    for i in range(0, k):
        for j in range(0, k):
            if i == j:
                continue

            testData = splitData[i]
            validationData = splitData[j]

            trainingFolds = splitData[:]
            del trainingFolds[i]
            # handles the case where i < j, so it changes the index of j
            if i < j:
                del trainingFolds[j-1]
            else:
                del trainingFolds[j]

            trainingData = np.concatenate(trainingFolds)

            tree = decision_tree_learning(np.transpose(trainingData))

            prune(tree, validationData)
            confusionMatrix, statistics, accuracy = evaluate(tree, testData)

            totalConfusionMatrix += confusionMatrix
            totalStatistics += statistics
            totalAccuracy += accuracy
            count += 1

            unpruned_tree_output_file = f'../out/unpruned/tree{i}_{j}.png'
            pruned_tree_output_file = f'../out/pruned/tree{i}_{j}.png'

            plot_dtree(tree, unpruned_tree_output_file)
            plot_dtree(tree, pruned_tree_output_file)

    # divide by number of test sets
    totalConfusionMatrix /= count
    totalStatistics /= count
    totalAccuracy /= count

    print(totalConfusionMatrix)
    print(totalStatistics)
    print(totalAccuracy)

    return (totalConfusionMatrix, totalStatistics, totalAccuracy)


def main(input_file_path):
    data = np.loadtxt(input_file_path, dtype='float')

    print("Unpruned:\n")
    cross_validate(data)
    print("Pruned:\n")
    cross_validate_pruning(data)

    return 0


if __name__ == "__main__":
    args = sys.argv
    main(args[1])
