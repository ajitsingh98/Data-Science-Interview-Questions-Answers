# Data Science Interview Questions And Answers

# Computer Science Questions

Contents
---

- [Algorithms](#algorithms)
- [Complexity and Numerical Analysis](#complexity-and-numerical-analysis)

---

## Algorithms

Q. Write a Python function to recursively read a JSON file.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Implement an  $O(NlogN)$  sorting algorithm, preferably quick sort or merge sort.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Find the longest increasing subsequence in a string.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Find the longest common subsequence between two strings.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Traverse a tree in pre-order, in-order, and post-order.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Given an array of integers and an integer $k$, find the total number of continuous subarrays whose sum equals $k$ . The solution should have $O(N)$  runtime.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. There are two sorted arrays nums1  and nums2  with m  and n  elements respectively. Find the median of the two sorted arrays. The solution should have $O(log(m+n))$  runtime.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Write a program to solve a Sudoku puzzle by filling the empty cells. The board is of the size  $9×9$ . It contains only $1-9$ numbers. Empty cells are denoted with *. Each board has one unique solution.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Given a memory block represented by an empty array, write a program to manage the dynamic allocation of that memory block. The program should support two methods: `malloc()` to allocate memory and `free()` to free a memory block.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Given a string of mathematical expression, such as `10 * 4 + (4 + 3) / (2 - 1)`, calculate it. It should support four operators `+`, `-`, `:`, `/`, and the brackets `()`.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Given a directory path, descend into that directory and find all the files with duplicated content.

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. In Google Docs, you have the `Justify alignment` option that spaces your text to align with both left and right margins. Write a function to print out a given text line-by-line (except the last line) in Justify alignment format. The length of a line should be configurable.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You’re free to choose a metric to define the “similarity” of content between files.

<details><summary><b>Answer</b></summary>
    
</details>

---

## Complexity and numerical analysis

Q. Matrix multiplication
1. You have three matrices: $A∈R^{100×5},B∈R^{5×200},C∈R^{200×20}$  and you need to calculate the product ABC . In what order would you perform your multiplication and why?
1. Now you need to calculate the product of $N$  matrices $A_1A_2...A_n$. How would you determine the order in which to perform the multiplication?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. What are some of the causes for numerical instability in deep learning?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. In many machine learning techniques (e.g. batch norm), we often see a small term  $ϵ$  added to the calculation. What’s the purpose of that term?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. What made GPUs popular for deep learning? How are they compared to TPUs?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. What does it mean when we say a problem is intractable?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. What are the time and space complexity for doing backpropagation on a recurrent neural network?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Is knowing a model’s architecture and its hyperparameters enough to calculate the memory requirements for that model?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Your model works fine on a single GPU but gives poor results when you train it on 8 GPUs. What might be the cause of this? What would you do to address it?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. What benefits do we get from reducing the precision of our model? What problems might we run into? How to solve these problems?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. How to calculate the average of 1M floating-point numbers with minimal loss of precision?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. How should we implement batch normalization if a batch is spread out over multiple GPUs?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Given the following code snippet. What might be a problem with it? How would you improve it?

```python
    def within_radius(a, b, radius):
        if np.linalg.norm(a - b) < radius:
            return 1
        return 0

    def make_mask(volume, roi, radius):
        mask = np.zeros(volume.shape)
        for x in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                for z in range(volume.shape[2]):
                    mask[x, y, z] = within_radius((x, y, z), roi, radius)
        return mask
```

<details><summary><b>Answer</b></summary>
    
</details>

---