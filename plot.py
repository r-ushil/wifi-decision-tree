import matplotlib.pyplot as plt

HEIGHT_OFFSET = 5
WIDTH_OFFSET = 50

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


def plot_tree_node(node: TreeNode, x: int):

  y = node.depth * HEIGHT_OFFSET

  # is a branch node
  if isinstance(node.label, Decision):
    label = 'E' + str(node.label.emitter) + ' < ' + str(node.label.value)
    plt.text(x, y, label)
    plot_tree_node(node.left, x - WIDTH_OFFSET)
    plot_tree_node(node.right, x + WIDTH_OFFSET)

  else:
    label = "Leaf: " + str(node.label)
    plt.text(x, y, label)

  


def plot_decision_tree(tree: TreeNode, max_depth: int):

  max_y = HEIGHT_OFFSET * max_depth
  max_x = WIDTH_OFFSET * (max_depth ** 2)

  plot_tree_node(tree, 0)

  ax = plt.gca()
  ax.set_xlim([-max_x, max_x])
  ax.set_ylim([0, max_y])

  plt.xticks([])
  plt.yticks([])
  plt.plot()
  plt.savefig('tree.png', format="png")
  plt.show()