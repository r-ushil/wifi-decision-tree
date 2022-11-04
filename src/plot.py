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

    def is_leaf(self):
        return self.room is not None

    def get_room(self, strengths: np.array) -> Room:
        # Base case: aready at a leaf node
        if self.is_leaf():
            return self.room

        # Otherwise, recurse based on decision.
        tree = self.left if strengths[self.decision.emitter] < self.decision.value else self.right

        return tree.get_room(strengths)

    

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
