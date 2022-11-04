from typing import Union
import matplotlib.pyplot as plt
import numpy as np

HEIGHT_OFFSET = 5
WIDTH_OFFSET = 50

TreeNodeData = Union['Decision', 'Room']

class Decision:
    def __init__(self, emitter: int, value):
        self.emitter = emitter
        self.value = value

    def get_label_txt(self):
        return f"E{self.emitter} < {self.value}"

class Room:
    def __init__(self, label: int, occurences: int):
        self.label = label
        self.occurences = occurences

    def get_label_txt(self):
        return f"Leaf: {self.label}"


class TreeNode:
    value: TreeNodeData

    def __init__(self, roomOrDecision, left=None, right=None):
        if left is None:
            # node represents a room
            self.room: Room = roomOrDecision
            self.decision = None
        else:
            # node represents a decision
            self.room: Room = None
            self.decision = roomOrDecision

        self.value = roomOrDecision

        self.left = left
        self.right = right

        # Maximum depth
        self.depth = 1 + max(
            left.depth if left else 0,
            right.depth if right else 0
        )

    def get_label_txt(self):
        return self.value.get_label_txt()

    def is_leaf(self):
        return self.room is not None

    def get_room(self, strengths: np.array) -> Room:
        # Base case: aready at a leaf node
        if self.is_leaf():
            return self.room

        # Otherwise, recurse based on decision.
        tree = self.left if strengths[self.decision.emitter] < self.decision.value else self.right

        return tree.get_room(strengths)

    def get_depth(self):
        l_depth = self.left.get_depth() if isinstance(self.left, TreeNode) else 0
        r_depth = self.right.get_depth() if isinstance(self.right, TreeNode) else 0

        return 1 + max(l_depth, r_depth)

    def count_nodes(self):
        l_node_count = self.left.count_nodes() if isinstance(self.left, TreeNode) else 0
        r_node_count = self.right.count_nodes() if isinstance(self.right, TreeNode) else 0

        return 1 + l_node_count + r_node_count

PADDING_X = 10
PADDING_Y = 10
LABEL_BBOX_STYLE = {
    'facecolor': 'white',
    'alpha': 1,
    'edgecolor': 'black',
    'linewidth': 0.5,
    'pad': 1,
}

def _plot_dtree_node(node: TreeNode, min_x: float) -> tuple[float, tuple[float, float]]:
    (lb_max_x, (lb_root_x, lb_root_y)) = _plot_dtree_node(node.left, min_x) if not node.is_leaf() else (min_x, (None, None))
    (root_x, root_y) = (lb_max_x + PADDING_X, node.depth * PADDING_Y)

    plt.text(root_x, root_y, node.get_label_txt(), bbox=LABEL_BBOX_STYLE, ha='center', va='center')

    (rb_max_x, (rb_root_x, rb_root_y)) = _plot_dtree_node(node.right, root_x) if not node.is_leaf() else (root_x, (None, None))

    if not node.is_leaf():
        plt.arrow(root_x, root_y, lb_root_x - root_x, lb_root_y - root_y)
        plt.arrow(root_x, root_y, rb_root_x - root_x, rb_root_y - root_y)

    return (rb_max_x, (root_x, root_y))

def plot_dtree(tree: TreeNode, filename: str):
    plt.figure()

    font_size = 100 / tree.count_nodes()
    font_size_dpi = max(600 / font_size, 180)

    plt.rcParams.update({'font.size': font_size })

    (max_x, _) = _plot_dtree_node(tree, 0)
    max_y = (tree.get_depth() - 1) * PADDING_Y

    amin_x = -0.1 * max_x
    amax_x = 1.1 * max_x

    amin_y = 0
    amax_y = max_y * 1.2

    ax = plt.gca()
    ax.set_xlim([amin_x, amax_x])
    ax.set_ylim([amin_y, amax_y])

    print(f"  Dims ({max_x}, {max_y})")

    plt.xticks([])
    plt.yticks([])
    plt.plot()
    plt.savefig(filename, format='png', dpi=font_size_dpi)
    plt.close()

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
