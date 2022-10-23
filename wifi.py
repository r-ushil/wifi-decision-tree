from cProfile import label
import numpy as np

LABEL_INDEX: int = 7



class Decision:

  def __init__(self, emitter: int, value: int):
    self.emitter = emitter
    self.value = value

class TreeNode:

  def __init__(self, label: Decision | int, depth: int):
    self.left = None
    self.right = None
    self.label = label
    self.depth = depth




def main():

  clean_dataset = np.loadtxt("./WIFI_db/clean_dataset.txt", dtype='int')
  (tree, depth) = decision_tree_learning(clean_dataset, 0)
  print(depth)
  return 0


def decision_tree_learning(dataset: np.ndarray, depth: int):

  labelCol = np.transpose(dataset)[LABEL_INDEX]

  # if all samples in dataset have same label (col 7)
  if np.all(labelCol[0] == labelCol):
    return (TreeNode(labelCol[0], depth), depth)

  else:

    (decision, leftData, rightData) = find_split(dataset)
     
    node = TreeNode(decision, depth)

    leftBranch, leftDepth = decision_tree_learning(leftData, depth+1)
    rightBranch, rightDepth = decision_tree_learning(rightData, depth+1)

    node.left = leftBranch
    node.right = rightBranch

    return (node, max(leftDepth, rightDepth))


def find_split(dataset: np.ndarray):

  # loop through every attribute, find split point and highest info gain
  # per attribute

  highestIG = 0
  splitVal = 0
  bestAttr = -1
  bestLeftSplit = bestRightSplit = [[]]

  datasetT = np.transpose(dataset)

  for i in range(LABEL_INDEX):
    (ig, midpoint, leftData, rightData) = split_attribute(datasetT, i)

    if ig > highestIG:

      highestIG = ig

      bestAttr = i
      splitVal = midpoint
      bestLeftSplit = leftData
      bestRightSplit = rightData

  decision = Decision(bestAttr, splitVal)


  return (decision, np.transpose(bestLeftSplit), np.transpose(bestRightSplit))
  

# takes in 2 row array (row 0: attribute vals, row 1: labels)
def split_attribute(dataset: np.ndarray, attributeNo: int):

  # (highestIG, midpoint, leftData, rightData)
  res = (0, 0, [[]], [[]])
  highestIG = 0

  # sort attribute by value (first row)
  sortedAttr = dataset[:, dataset[attributeNo].argsort()]

  noOfExamples = np.shape(dataset)[1]

  for i in range(noOfExamples - 1):

    # sliding window of size 2
    valPair = sortedAttr[attributeNo][i:i+2]

    midpoint = (valPair[0] + valPair[1]) / 2

    # slice sorted 2d array into two parts, seperated by i
    leftData = sortedAttr[:, :i+1]
    rightData = sortedAttr[:, i+1:]

    currIG = infoGain(sortedAttr[LABEL_INDEX], leftData[LABEL_INDEX], rightData[LABEL_INDEX])

    # update result var if new IG is higher
    if currIG > highestIG:
      highestIG = currIG
      res = (highestIG, midpoint, leftData, rightData)

  return res

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


def remainder(leftLabels: np.array, rightLabels: np.array):

  leftTotal = np.shape(leftLabels)[0]
  rightTotal = np.shape(rightLabels)[0]
  combinedTotal = leftTotal + rightTotal

  leftRem = (leftTotal / combinedTotal) * entropy(leftLabels)
  rightRem = (rightTotal / combinedTotal) * entropy(rightLabels)

  return leftRem + rightRem


def infoGain(datasetLabels, leftLabels, rightLabels):
  return entropy(datasetLabels) - remainder(leftLabels, rightLabels)


if __name__ == "__main__":
  main()