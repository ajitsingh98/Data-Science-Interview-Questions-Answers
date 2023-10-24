1. RNNs
    1. What’s the motivation for RNN?
    1. What’s the motivation for LSTM?
    1. How would you do dropouts in an RNN?
---
2. What’s density estimation? Why do we say a language model is a density estimator?

---

3. Language models are often referred to as unsupervised learning, but some say its mechanism isn’t that different from supervised learning. What are your thoughts?

---

4. Word embeddings.
    1. Why do we need word embeddings?
    1. What’s the difference between count-based and prediction-based word embeddings?
    1. Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?

---

5. Given 5 documents:
     
    ```
     D1: The duck loves to eat the worm
     D2: The worm doesn’t like the early bird
     D3: The bird loves to get up early to get the worm
     D4: The bird gets the worm from the early duck
     D5: The duck and the birds are so different from each other but one thing they have in common is that they both get the worm
    ```
      1. Given a query Q: “The early bird gets the worm”, find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}. Are the top-ranked documents relevant to the query?
      2. Assume that document D5 goes on to tell more about the duck and the bird and mentions “bird” three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?

---

6. Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?

---

7. For n-gram language models, does increasing the context length (n) improve the model’s performance? Why or why not?

---

8. What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?

---
9. What's the Levenshtein distance of the two words “doctor” and “bottle”?

---

10. BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?

---

11. On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?

---
12. Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?

---

13. Why does removing stop words sometimes hurt a sentiment analysis model?

---

14. Many models use relative position embedding instead of absolute position embedding. Why is that?

---

15. Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What’s the purpose of this?

---
