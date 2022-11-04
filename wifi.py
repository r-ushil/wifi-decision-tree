from typing import Tuple
import numpy as np

from plot import Room, plot_decision_tree, Decision, TreeNode

LABEL_INDEX: int = 7
NUM_ROOMS: int = 4


def main():
    clean_data = np.loadtxt("./WIFI_db/clean_dataset.txt", dtype='float')
    noisy_data = np.loadtxt("./WIFI_db/noisy_dataset.txt", dtype='float')

    print("Clean data\n")
    print("Unpruned\n")
    cross_validate(clean_data)
    print("Pruned\n")
    cross_validate_pruning(clean_data)

    print("Noisy data\n")
    print("Unpruned\n")
    cross_validate(noisy_data)
    print("Pruned\n")
    cross_validate_pruning(noisy_data)
    
    return 0


def decision_tree_learning(dataT: np.ndarray):
    labels = dataT[LABEL_INDEX]

    # If all labels are the same, no more decisions to be made.
    # Return a leaf node with that label and no of occurences of that label.
    if np.all(labels[0] == labels):
        return TreeNode(Room(int(labels[0]), np.shape(labels)[0]))

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

    np.random.seed(42) # for reproducibility
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

    # divide by number of test sets
    totalConfusionMatrix /= k
    totalStatistics /= k
    totalAccuracy /= k

    print(totalConfusionMatrix)
    print(totalStatistics)
    print(totalAccuracy)

    return (totalConfusionMatrix, totalStatistics)

def confusion_matrix(tree: TreeNode, testData: np.ndarray):
    # Makes a LABEL_INDEX * LABEL_INDEX confusion matrix
    confusionMatrix = np.zeros((NUM_ROOMS, NUM_ROOMS))
    # confusionMatrix[n - 1][m - 1] = number of times the phone was in room n,
    # and we predicted it was in room m

    for [*strengths, actualRoom] in testData:
        predictedRoom = tree.get_room(strengths).label
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
        totals[roomIdx][3] += np.sum(confusionMatrix[:, roomIdx]) - truePositive
    return confusionMatrix, totals



def evaluate(tree: TreeNode, testData: np.ndarray):
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
        statistics[roomIdx][3] = 2 * (statistics[roomIdx][0] * statistics[roomIdx][1]) / (statistics[roomIdx][0] + statistics[roomIdx][1])
        
    avgAccuracy = np.sum(statistics[:, 2]) / NUM_ROOMS
    return (confusionMatrix, statistics, avgAccuracy)


def prune_accuracy(node: TreeNode, validationData: np.ndarray):

    # if there is no more validation data, we prune anyway
    if validationData.size == 0:
        return float('inf')
    
    if node.is_leaf():
        roomIdx = node.room.label - 1
        (_, room_totals) = confusion_matrix(node, validationData)
        
        tp = room_totals[roomIdx][0]
        tn = room_totals[roomIdx][1]
        fp = room_totals[roomIdx][2]
        fn = room_totals[roomIdx][3]

        return (tp + tn) / (tp + tn + fp + fn)


    leftValidation, rightValidation = split_validation(node, validationData)
    leftAccuracy = prune_accuracy(node.left, leftValidation)
    rightAccuracy = prune_accuracy(node.right, rightValidation)

    return (leftAccuracy + rightAccuracy) / 2


def prune(tree: TreeNode, validation_data: np.ndarray):

    while True:
        (prunedTree, pruned) = prune_once(tree, validation_data)
        if not pruned: break
    return prunedTree

def prune_once(tree: TreeNode, validation_data: np.ndarray):
    if tree.is_leaf():
            return (tree, False)

    if tree.left.is_leaf() and tree.right.is_leaf():
        leafReplacingNode = tree.left if tree.left.room.occurences > tree.right.room.occurences else tree.right
        
        leafAccuracy = prune_accuracy(leafReplacingNode, validation_data)
        treeAccuracy = prune_accuracy(tree, validation_data)

        if leafAccuracy >= treeAccuracy:
            return (leafReplacingNode, True)
        else:
            return (tree, False)

    leftValidation, rightValidation = split_validation(tree, validation_data)
    (tree.left, leftPruned) = prune_once(tree.left, leftValidation[:])
    (tree.right, rightPruned) = prune_once(tree.right, rightValidation[:])

    return (tree, (leftPruned or rightPruned))



# PRE: tree is not a leaf
def split_validation(tree: TreeNode, validation_data: np.ndarray):
    assert(tree.left is not None and tree.right is not None)

    col = tree.decision.emitter
    val = tree.decision.value
    data = validation_data

    l_validation_data = data[data[:, col] < val]
    r_validation_data = data[data[:, col] >= val]

    return l_validation_data, r_validation_data
    

# data is NOT transpose (each column is an emitter)
# WARNING: mutates data when shuffling
def cross_validate_pruning(data: np.ndarray):
    k = 10

    np.random.seed(42) # for reproducibility
    np.random.shuffle(data)

    splitData = np.split(data, k)

    totalConfusionMatrix = np.zeros((NUM_ROOMS, NUM_ROOMS))
    totalStatistics = np.zeros((NUM_ROOMS, 4))
    totalAccuracy = 0
    count = 0

    for i in range(0, k):
        for j in range(0, k):
            if i==j: continue

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
            prunedTree = prune(tree, validationData)
            confusionMatrix, statistics, accuracy = evaluate(prunedTree, testData)
            
            totalConfusionMatrix += confusionMatrix
            totalStatistics += statistics
            totalAccuracy += accuracy
            count += 1

    # divide by number of test sets
    totalConfusionMatrix /= count
    totalStatistics /= count
    totalAccuracy /= count

    print(totalConfusionMatrix)
    print(totalStatistics)
    print(totalAccuracy)

    return (totalConfusionMatrix, totalStatistics, totalAccuracy)

if __name__ == "__main__":
    main()
