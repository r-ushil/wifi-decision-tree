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

  decision = Decision(0, 0)
  return (decision, 0, 0)
  


if __name__ == "__main__":
  main()