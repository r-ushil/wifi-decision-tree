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
  decision_tree_learning(clean_dataset, 0)
  return 0


def decision_tree_learning(dataset: np.ndarray, depth: int):

  labelCol = dataset[LABEL_INDEX]

  # if all samples in dataset have same label (col 7)
  if np.all(labelCol[0] == labelCol):
    return TreeNode(labelCol[0], depth)

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

  # keep track of split point and attribute

  decision = Decision(0, 0)
  return (decision, 0, 0)
  

# takes in 2 row array (row 0: attribute vals, row 1: labels)
def split_attribute(attribute: np.ndarray) -> tuple(int, np.ndarray, np.ndarray):

  LABEL_INDEX = 1

  # (midpoint, leftData, rightData)
  res = (0, [[]], [[]])
  highestIG = 0

  # sort attribute by value (first row)
  sortedAttr = attribute[:, attribute[0].argsort()]

  # loop through midpoints for all adjacent vals, sliding window?

  noOfAttr = np.shape(attribute)[0]

  for i in range(noOfAttr - 1):

    # sliding window of size 2
    valPair = attribute[0][i:i+2]
    midpoint = (valPair[0] + valPair[1]) / 2

    # slice sorted 2d array into two parts, seperated by i
    leftData = attribute[0:1, 0:i]
    rightData = attribute[0:1, i+1:]

    currIG = infoGain(attribute[LABEL_INDEX], leftData[LABEL_INDEX], rightData[LABEL_INDEX])
    
    # update result var if new IG is higher
    if currIG > highestIG:
      res = (midpoint, leftData, rightData)

  return res

def entropy(labels: np.array):

  # as labels array only has values from 1-4, bincount and cut off 0
  occurences = np.bincount(labels)[1:]
  
  totalExamples = np.shape(labels)[0]

  probabilities = occurences / totalExamples

  entropy = 0
  for p in np.nditer(probabilities):
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