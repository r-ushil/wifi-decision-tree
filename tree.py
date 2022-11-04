import numpy as np


class Decision:
    def __init__(self, emitter: int, value):
        self.emitter = emitter
        self.value = value

    def get_label_txt(self):
        return f"E[{self.emitter}] < {self.value}"


class TreeLeaf:
    def __init__(self, room: int, roomCounts):
        self.room = room
        self.roomCounts = roomCounts

        self.avg_depth = 1
        self.max_depth = 1
        self.size = 1

    def is_leaf(self):
        return True

    def get_room(self, strengths):
        return self.room

    def merge_leaves(self, other):
        roomCounts = dict(self.roomCounts)
        for k, v in other.roomCounts.items():
            if k not in roomCounts:
                roomCounts[k] = 0

            roomCounts[k] += v

        room = max(roomCounts, key=roomCounts.get)

        return TreeLeaf(room, roomCounts)

    def get_label_txt(self):
        return f"{self.room}"


class TreeBranch:
    def __init__(self, decision, left, right):
        self.decision = decision
        self.left = left
        self.right = right

        # Stats
        self.avg_depth = 1 + (self.left.avg_depth + self.right.avg_depth) / 2
        self.max_depth = 1 + max(self.left.max_depth, self.right.max_depth)
        self.size = 1 + self.left.size + self.right.size

    def is_leaf(self):
        return False

    def get_room(self, strengths: np.array) -> int:
        # Recurse based on decision.
        tree = self.left if strengths[self.decision.emitter] < self.decision.value else self.right
        return tree.get_room(strengths)

    def get_label_txt(self):
        return f"{self.decision.get_label_txt()}"
