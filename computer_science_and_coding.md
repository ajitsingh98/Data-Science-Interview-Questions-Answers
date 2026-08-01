# Computer Science & Coding Interview Questions and Answers

## Table of Contents
- [Algorithms & Data Structures](#algorithms--data-structures)
- [Complexity & Numerical Analysis](#complexity--numerical-analysis)

---

## Algorithms & Data Structures

### Q: Write a Python function to recursively read and traverse a JSON file or nested dictionary.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
import json

def traverse_json(data, prefix=""):
    """Recursively traverse a nested JSON structure (dicts and lists)."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            traverse_json(value, new_key)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_key = f"{prefix}[{idx}]"
            traverse_json(item, new_key)
    else:
        print(f"{prefix}: {data}")

# Example Usage:
# with open('data.json') as f:
#     data = json.load(f)
#     traverse_json(data)
```

</details>

---

### Q: Implement an $O(N \log N)$  sorting algorithm (QuickSort or MergeSort in Python).

<details>
<summary><b>💡 Show Answer</b></summary>

```python
def quicksort(arr):
    """In-place QuickSort algorithm with O(N log N) average time complexity."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Test
print(quicksort([3, 6, 8, 10, 1, 2, 1]))
# Output: [1, 1, 2, 3, 6, 8, 10]
```

</details>

---

### Q: Find the length of the Longest Increasing Subsequence (LIS) in an array/string.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
import bisect

def length_of_lis(nums):
    """
    Returns the length of LIS using Patient Sorting & Binary Search in O(N log N) time.
    """
    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    return len(tails)

# Example
print(length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]))  # Output: 4 ([2, 3, 7, 101])
```

</details>

---

### Q: Find the Longest Common Subsequence (LCS) between two strings.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Dynamic Programming approach with O(M * N) time and O(M * N) space complexity.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Example
print(longest_common_subsequence("abcde", "ace"))  # Output: 3 ("ace")
```

</details>

---

### Q: Traverse a Binary Tree in Pre-Order, In-Order, and Post-Order (Iterative or Recursive).

<details>
<summary><b>💡 Show Answer</b></summary>

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pre_order(root):
    return [root.val] + pre_order(root.left) + pre_order(root.right) if root else []

def in_order(root):
    return in_order(root.left) + [root.val] + in_order(root.right) if root else []

def post_order(root):
    return post_order(root.left) + post_order(root.right) + [root.val] if root else []
```

</details>

---

### Q: Given an array of integers and an integer $k $, find the total number of continuous subarrays whose sum equals $ k $in$ O(N)$  runtime.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
def subarray_sum(nums, k):
    """
    Uses Prefix Sums + Hash Map to find count of subarrays with sum k in O(N) time.
    """
    count = 0
    current_sum = 0
    prefix_sums = {0: 1}  # Base case: sum of 0 appears once

    for num in nums:
        current_sum += num
        if current_sum - k in prefix_sums:
            count += prefix_sums[current_sum - k]
        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1

    return count

# Example
print(subarray_sum([1, 1, 1], 2))  # Output: 2
```

</details>

---

### Q: Find the Median of Two Sorted Arrays in $O(\log(m+n))$  runtime.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
def find_median_sorted_arrays(nums1, nums2):
    """Binary Search partition method in O(log(min(m, n))) time."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    x, y = len(nums1), len(nums2)
    low, high = 0, x

    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (x + y + 1) // 2 - partitionX

        maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
        minX = float('inf') if partitionX == x else nums1[partitionX]

        maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
        minY = float('inf') if partitionY == y else nums2[partitionY]

        if maxX <= minY and maxY <= minX:
            if (x + y) % 2 == 0:
                return (max(maxX, maxY) + min(minX, minY)) / 2.0
            else:
                return max(maxX, maxY)
        elif maxX > minY:
            high = partitionX - 1
        else:
            low = partitionX + 1
```

</details>

---

### Q: Write a program to solve a Sudoku puzzle using Backtracking.

<details>
<summary><b>💡 Show Answer</b></summary>

```python
def solve_sudoku(board):
    """Solves 9x9 Sudoku board in-place using Depth-First Search Backtracking."""
    def is_valid(r, c, val):
        for i in range(9):
            if board[r][i] == val or board[i][c] == val:
                return False
            if board[3 * (r // 3) + i // 3][3 * (c // 3) + i % 3] == val:
                return False
        return True

    for r in range(9):
        for c in range(9):
            if board[r][c] == '.':
                for num in map(str, range(1, 10)):
                    if is_valid(r, c, num):
                        board[r][c] = num
                        if solve_sudoku(board):
                            return True
                        board[r][c] = '.'
                return False
    return True
```

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)