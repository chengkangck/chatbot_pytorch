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

### 1.3 Project flow

- Data preprocessing

- Data loading

- Model building

- Training model

- Test model

```
data
  qingyun.tsv   # Corpus
model
  model.py      #seq2seq model
  pre processing.py #data preproessing
requirements.txt
train.py
text.py
demo.py
modelA
modelB
modelC
modelD
```

## 2 For research

### 2.1 import the module
```
from model.nnModel import *
from model.corpusSolver import *
import torch
```
### 2.2 how to load the data
```
dataClass = Corpus('./corpus/qingyun.tsv', maxSentenceWordsNum=25)
# First parameter is your corpus path;
# maxSentenceWordsNum will ignore the data whose words number of question or answer is too big;
```
Also you can load your corpus. Only your file content formats need to be consistent:
```
Q A
Q A
...
```
Every line is a question and a answer with a '\t' split.
The corpus comes from https://github.com/codemayq/chinese_chatbot_corpus, you can get larger corpus from this Github.

### 2.3 How to train your model
First you need to create a Seq2Seq object.

```
model = Seq2Seq(dataClass, featureSize=256, hiddenSize=256, 
                attnType='L', attnMethod='general', 
                encoderNumLayers=3, decoderNumLayers=2, 
                encoderBidirectional=True, 
                device=torch.device('cuda:0'))


First parameter is your corpus class object.
featureSize is your word vector size;
hiddenSize is your RNN hidden state size;
attnType is your attention type. It can be 'B' for using Bahdanau Attention Structure or 'L' for using Luong Structure;
attnMethod is Luong Attention Method. It can be 'dot', 'general' or 'concat'.
encoderNumLayers is the layer number of your encoder RNN;
decoderNumlayers is the layer number of your decoder RNN;
encoderBidirectional is if your encoder RNN is bidirectional;
device is your building environment. If using CPU, then device=torch.device('cpu'); if using GPU, then device=torch.device('cuda:0');
```
Then you can train your model.

```
model.train(batchSize=1024, epoch=500)

```
```

batchSize is the number of data used for each train step;
epoch is the total iteration number of your training data;

```
And the log will be print like follows:

```
...
After iters 6540: loss = 0.844; train bleu: 0.746, embAve: 0.754; 2034.582 qa/s; remaining time: 48096.110s;
After iters 6550: loss = 0.951; train bleu: 0.734, embAve: 0.755; 2034.518 qa/s; remaining time: 48092.589s;
After iters 6560: loss = 1.394; train bleu: 0.735, embAve: 0.759; 2034.494 qa/s; remaining time: 48088.128s;
...
```
Finally you need to save your model for future use.
```
model.save('model.pkl')
First parameter is the name of model saved.

```
Ok, I know you are too lazy to train your own model. Also you can download my trained model. I will upload my model to the cloud and share the link and password.
![image](https://github.com/chengkangck/chatbot_pytorch/blob/main/images/modeltrained.PNG)

### 2.4 How to use your model to build a chatbot
First you need to create a Chatbot object.
```
chatbot = Chatbot('model.pkl')
First parameter is your model path;

```
Then you can use the greedy search to generate the answer.
chatbot.predictByGreedySearch("你好啊")
First parameter is your question;

It will return the answer like "你好,我就开心了". Also you can plot the attention by showAttention=True. Or you can use the beam search to generate the answer.

```
chatbot.predictByBeamSearch("什么是ai", isRandomChoose=True, beamWidth=10)
First parameter is your question;
isRandomChoose determines whether probability sampling is performed in the final beamwidth answers.
beamWidth is the search width in beam search;
```
It will return the answer like "反正不是苹果". Also you can show the probabilities of the beamwidth answers by showInfo=True.

### 2.5 How to use a trained word embedding
First you need to calculate 4 variables:
```
id2word: a list of word, and the first two words have to be "<SOS>" and "<EOS>", e.g., ["<SOS>", "<EOS>", "你", "天空", "人工智能", "中国", ...];
word2id: a dict with the key of word and the value of id, corresponding to id2word, e.g., {"<SOS>":0, "<EOS>":1, "你":2, "天空":3, "人工智能":4, "中国":5, ...};
wordNum: the total number of words. It is equal to len(id2word) or len(word2id);
wordEmb: the word embedding array with shape (wordNum, featureSize) and the word order need to be consistent with id2word or word2id; you can random initialize the vector or "<SOS>" and "<EOS>";

```
Then add first three variables as parameters when you load the data.

```
dataClass = Corpus(..., id2word=id2word, word2id=word2id, wordNum=wordNum)
```

Next you need to create the word embedding object.
```
embedding = torch.nn.Embedding.from_pretrained(torch.tensor(wordEmb))
```
Finally add the embedding parameter when you create the Seq2Seq object.

```
model = Seq2Seq(..., embedding=embedding)
```
Also you can download a trained word embedding from https://github.com/Embedding/Chinese-Word-Vectors.
Thanks very much for the trained word embedding provided by the author.

### Future work
For other functions such as data enhance, etc, please dig for yourselves.



The learning material fot the seq2seq_machine translation_detoen.ipynb. https://www.cnblogs.com/jfdwd/p/11090382.html  Pytorch learning records- torchtext and Pytorch examples (using neural network Seq2Seq code)

