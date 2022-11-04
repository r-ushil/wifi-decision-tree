import numpy as np

class Decision:
    def __init__(self, emitter: int, value):
        self.emitter = emitter
        self.value = value

class TreeLeaf:
    def __init__(self, room: int, roomCounts):
        self.room = room
        self.roomCounts = roomCounts
        self.depth = 1

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

class TreeBranch:
    def __init__(self, decision, left, right):
        self.decision = decision
        self.left = left
        self.right = right

        # Maximum depth
        self.depth = 1 + max(
            self.left.depth if self.left else 0,
            self.right.depth if self.right else 0
        )

    def is_leaf(self):
        return False

    def get_room(self, strengths: np.array) -> int:
        # Recurse based on decision.
        tree = self.left if strengths[self.decision.emitter] < self.decision.value else self.right
        return tree.get_room(strengths)
