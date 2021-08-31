# chatbot_pytorch_seq2seq+attention

## 1. Project introduction

This project focuses on the mainstream direction of NLP, uses the Pytorch framework, and uses the mainstream technology Seq2seq+Attention to develop generative chatbot.

### 1.1 Data source
Every line is a question and a answer with a '\t' split.
The corpus comes from https://github.com/codemayq/chinese_chatbot_corpus, you can get larger corpus from this Github.
Thanks very much for the corpus summarized and processed by the author.
![image](https://github.com/chengkangck/chatbot_pytorch/blob/main/images/data%20source%20example.PNG)

### 1.2 model introduction
![image](https://github.com/chengkangck/chatbot_pytorch/blob/main/images/model1.PNG)
![image](https://github.com/chengkangck/chatbot_pytorch/blob/main/images/seq2seq.png)

**Attention** :

- [1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.
- [2] Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[J]. arXiv preprint arXiv:1508.04025, 2015.

## https://www.cnblogs.com/jfdwd/p/11090382.html  Pytorch learning records- torchtext and Pytorch examples (using neural network Seq2Seq code)

纸上得来终觉浅，绝知此事要躬行
