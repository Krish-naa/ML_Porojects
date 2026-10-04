"""
232. Implement Queue using Stacks
https://leetcode.com/problems/implement-queue-using-stacks/

Implement a first-in-first-out (FIFO) queue using only two stacks.
The queue should support push, pop, peek, and empty.

Approach:
    Use two stacks, `in_stack` and `out_stack`.
    - push: always append to in_stack.
    - pop / peek: if out_stack is empty, pour everything from in_stack into
      out_stack (which reverses the order), then operate on out_stack's top.
    Each element is moved at most once from in_stack to out_stack, so the
    amortized time for each operation is O(1).
"""


class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self.peek()
        return self.out_stack.pop()

    def peek(self) -> int:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack


if __name__ == "__main__":
    q = MyQueue()
    q.push(1)
    q.push(2)
    print(q.peek())   # 1
    print(q.pop())    # 1
    print(q.empty())  # False
