import numpy as np

LABEL_INDEX: int = 7


class TreeNode:

  def __init__(self, label: bool | int, depth: int):
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

    return 0

  return 0


if __name__ == "__main__":
  main()