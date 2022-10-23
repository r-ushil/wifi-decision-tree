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

  entropy(np.transpose(dataset)[LABEL_INDEX])

  decision = Decision(0, 0)
  return (decision, 0, 0)
  


def entropy(labels: np.array):

  # as labels array only has values from 1-4, bincount and cut off 0
  occurences = np.bincount(labels)[1:]
  
  totalExamples = np.shape(labels)[0]

  probabilities = occurences / totalExamples

  entropy = 0
  for p in np.nditer(probabilities):
    entropy -= p * np.log2(p)
    
  return entropy


if __name__ == "__main__":
  main()