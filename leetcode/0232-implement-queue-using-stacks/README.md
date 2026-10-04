# 232. Implement Queue using Stacks

**Difficulty:** Easy
**Link:** https://leetcode.com/problems/implement-queue-using-stacks/

## Problem
Implement a FIFO queue using only two stacks. Support `push`, `pop`, `peek`, and `empty`.

## Approach
Keep two stacks:
- `in_stack` receives all new elements on `push`.
- `out_stack` serves `pop`/`peek`. When it's empty, pour everything from `in_stack`
  into `out_stack`, which reverses the order into correct FIFO order.

Each element moves from `in_stack` to `out_stack` at most once.

## Complexity
- **push:** O(1)
- **pop / peek:** O(1) amortized (O(n) worst case during a transfer)
- **empty:** O(1)
- **Space:** O(n)
