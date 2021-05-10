---
layout: post-deep-learning
title: Introduction to LSTMs
category: deep-learning
editor_options: 
  markdown: 
    wrap: 72
---

In this tutorial we will perform a sentiment analysis on movie reviews from the Movie Review Dataset popularly known as the IMDB dataset. In this task, given a review, the model attempts to predict whether the review was positive or negative.

Using this as an example we will learn about how a specific recurrent neural network (RNN) architecture known as Long Short-Term Memory (LSTM), designed to model temporal sequences and their long-range dependencies more accurately than the coventional RNNs can be implemented using KERAS.



In a coventional <a href='https://emmanuel-arize.github.io/datascience-blog/deeplearning/deep-learning/2021/05/06/RNN.html' target="_blank">  recurrent neural network</a>, during the backpropagation phase, in which the error signal (gradients) are backpropagated through time, the recurrent hidden layers (weight matrix associated with the layers) are subject to repeated multiplications as determined by as the number of timesteps (length of the sequence), so small weights tends to lead to a situation known as <b>vanishing gradients</b> where the error signal  propagating backwards gets so small that learning either becomes very slow or stops working altogether (error signals fowing backwards in time tend to vanish). Conversely larger weights tends to lead to a situation where the error signal is so large that it can cause learning to diverge , a situation known as <b>exploding gradients</b>.

To read more on exploding and vanishing gradients have a look at this papers
<br/>
<a href='https://arxiv.org/pdf/1211.5063v1.pdf' target="_blank">Understanding the exploding gradient problem</a><br/>
<a href='https://www.semanticscholar.org/paper/Learning-long-term-dependencies-with-gradient-is-Bengio-Simard/d0be39ee052d246ae99c082a565aba25b811be2d' target="_blank">Learning long-term dependencies with gradient descent is difficult</a><br/> 

<a href='https://www.bioinf.jku.at/publications/older/2304.pdf' target="_blank">THE VANISHING GRADIENT PROBLEM DURING LEARNING RECURRENT NEURAL NETS AND PROBLEM SOLUTIONS</a><br/>


The vanishing and exploding gradients problem, limit the ability of conventional RNNs in modeling sequences with long range contextual dependencies and to address these issues, more complex network architectures known as Gated Neural Networks (GNNs) have been designed to help mitigate this problem by introducing “gates”  to control the flow of information into and out of the  network layers. There are several GNNs but in this tutorial we were learn about a notable example known Long short-term memory (LSTM) networks (<a href='http://www.bioinf.jku.at/publications/older/2604.pdf' target='_blank'>Hochreiter and Schmidhuber, 1997</a>)

# Long Short-Term Memory NETWORKS (LSTMs)

The core idea of LSTMs is their memory cell state denoted by $$C_{t}$$ which is controled by the gating mechanism. At each time step, the controllable gating mechanisms decide which parts of the inputs will be written to the memory cell state, and which parts of memory cell state will be overwritten (forgotten), regulating information flowing into and out of the memory cell state, making LSTMs networks dividing the context management problem into two sub-problems: removing information no longer needed from the context and adding information likely to be needed for later decision making to the context. <a href='#lstm'>Figure 1</a>  is a A schematic diagram of LSTMs.
<br/>
<br/>
<img img id='lstm'  class="w3-center" src="{{'/assets/images/deep/keras/LSTM.png' |relative_url}}"><span id='Fig'>Figure 1</span>
<br/>
<a href='https://www.researchgate.net/figure/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell_fig5_329362532'>source <a/>

From <a href='#lstm'>Figure 1</a> the first step of the LSTM model is to decide  how to delete information from the context that is no longer needed and this is controlled by the
the <b>forget gate </b> denoted by $$f_t$$. The forgate gate is responsible for regulating the removal of information from the cell state that is no longer needed and is defined as


$$f_{t}=\sigma(x_{t}U^{f} +h_{t-1}W^{f} $$

$$k_{t}=c_{t-1} \odot f_{t}$$


where $$W^{f}$$ denotes the hidden to hidden weights with the superscript $$f$$ as a symbol indicating the forget gate, $$U^{f}$$ denoting input to hidden weights. The forget gate takes $$h_{t−1}$$, the previous hidden state and $$x_{t}$$, the input of the current time step t (time steps correspond to word positions in a sentence). The forget gate uses the logistic sigmoid activation function which output a vector with values between 0 and 1, with 0 representing a complete discarding of the information and 1 representing the keeping of information thereby adaptively resetting the cell memory. Through the forget gate, information which are of less importance are removed from the cell state. The vector output from the forget gate is then multiplied to the cell state

The next step is to compute the actual information (create a contextual vector $$C^{t}$$) needed to extract from the previous hidden state and current inputs defined by
$$\tilde{C_{t}} = tanh(U^{g}x_{t} + W^{g}h_{t−1} )$$

<b>NB:</b>
This is a contextual vector $$\tilde{C_{t}}$$ containing all possible values that needs to be
added to the cell state 

The next step is to decide the new information that will be stored in the cell state and is regulated by the <b> add aget</b>. The <b>add gate</b>  select the information to add to the
current context and is defined

$$i_{t} = \sigma(U^{i}x_{t} +W^{i}h_{t−1})$$

$$j_{t} = \tilde{C_{t}}\odot i_{t}$$

Next, we add this to the modified context vector to get our new context vector

$$C_{t}=k_{t}+j_{t}$$

<b>NB :</b> This is the Cell state that stores information and is responsible for remembering information for long period of time

Not all information stored in the cell state is required for the current hidden state, so the <b>output gate</b> decides what information is required for the current hidden state (as opposed to what information needs to be preserved for future decisions).

$$o_{t} = \sigma(U^{o}x_{t} +W^{o}h_{t−1})$$

$$h_{t}$$ is then defined as
$$h_{t}=o_{t} \odot tanh(C_{t})$$

<b>Below is an implemtation of LSTM using Keras</b>

loading the neede packages


```python
from tensorflow.keras.layers import Embedding,Layer,Flatten,Dense,Dropout,SimpleRNN,LSTM
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from tensorflow.keras.models import Sequential
import os,re,string
import numpy as np
```

# Loading the IMDB data 
You’ll restrict the movie reviews to the top 15,000 most common words and  considering looking at the first 30 words in every review. The network will learn 16-dimensional embeddings for each of the 15,000 words


```python
batch_size=100
seed = 100
tranin_data=tf.keras.preprocessing.text_dataset_from_directory(
                           directory='./data/aclImdb/train/',
                          batch_size=batch_size,subset='training',
                          validation_split=0.25,seed=seed)

val_data=tf.keras.preprocessing.text_dataset_from_directory(
                          directory='./data/aclImdb/train/',
                           batch_size=batch_size,subset='validation',
                            validation_split=0.25,seed=seed)
```

    Found 25000 files belonging to 2 classes.
    Using 18750 files for training.
    Found 25000 files belonging to 2 classes.
    Using 6250 files for validation.
    


```python
def remove_br(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')
```


```python
max_features = 15000  # Maximum vocab size.
max_tokens=228

encoded_input=TextVectorization(max_tokens=max_features,output_mode='int',
                                output_sequence_length=max_tokens,
                               standardize=remove_br)
```


```python
encoded_input.adapt(tranin_data.map(lambda x,y:x))
```


```python
embedded_dim=16
```


```python
model=Sequential()
model.add(encoded_input)
model.add(Embedding(input_dim=max_features,input_length=max_tokens,output_dim=embedded_dim))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
```


```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```


```python
history = model.fit(tranin_data,epochs=10,batch_size=150,validation_data=val_data)
```

    Epoch 1/10
    188/188 [==============================] - 65s 323ms/step - loss: 0.6921 - acc: 0.5041 - val_loss: 0.6245 - val_acc: 0.6787
    Epoch 2/10
    188/188 [==============================] - 50s 261ms/step - loss: 0.5307 - acc: 0.7618 - val_loss: 0.4656 - val_acc: 0.8123
    Epoch 3/10
    188/188 [==============================] - 50s 263ms/step - loss: 0.3776 - acc: 0.8485 - val_loss: 0.3668 - val_acc: 0.8672
    Epoch 4/10
    188/188 [==============================] - 51s 267ms/step - loss: 0.2959 - acc: 0.8915 - val_loss: 0.3502 - val_acc: 0.8653
    Epoch 5/10
    188/188 [==============================] - 51s 267ms/step - loss: 0.2571 - acc: 0.9175 - val_loss: 0.4001 - val_acc: 0.8672
    Epoch 6/10
    188/188 [==============================] - 51s 268ms/step - loss: 0.2284 - acc: 0.9219 - val_loss: 0.4741 - val_acc: 0.8045
    Epoch 7/10
    188/188 [==============================] - 51s 268ms/step - loss: 0.1997 - acc: 0.9363 - val_loss: 0.3648 - val_acc: 0.8758
    Epoch 8/10
    188/188 [==============================] - 51s 266ms/step - loss: 0.1754 - acc: 0.9476 - val_loss: 0.4051 - val_acc: 0.8741
    Epoch 9/10
    188/188 [==============================] - 51s 266ms/step - loss: 0.1666 - acc: 0.9475 - val_loss: 0.3162 - val_acc: 0.8760
    Epoch 10/10
    188/188 [==============================] - 52s 276ms/step - loss: 0.1435 - acc: 0.9579 - val_loss: 0.3463 - val_acc: 0.8662
    

# Testing the model on new inputs


```python
text = [
     "This movie is fantastic! I really like it because it is so good!",
    "Not a good movie!",
    "The movie was great!",
    "This movie really sucks! Can I get my money back please?",
    "The movie was terrible...",
    "This is a confused movie.",
  "The movie was great!",
  "This is a confused movie.",
  "The movie was terrible..."
    
]
for i in text:
    predictions = model.predict(np.array([i]))
    result='positive review' if predictions>0.5 else 'negative review'
    print(result)
```

    positive review
    negative review
    positive review
    negative review
    negative review
    negative review
    positive review
    negative review
    negative review
    

Reference

<a href='https://web.stanford.edu/~jurafsky/slp3/9.pdf'>Deep Learning Architectures
for Sequence Processing</a><br/>

<a href='https://www.deeplearningbook.org/contents/rnn.html'>
Ian Goodfellow, Yoshua Bengio and Aaron Courville (2016). Deep Learning. MIT Press,pp.389-413
</a>

<a href="https://link.springer.com/article/10.1007/BF00114844">Elman, J. L. (1991). Distributed representations, simple recurrent networks, and grammatical structure. Machine learning, 7(2), 195-225.</a><br/>

<a href="https://arxiv.org/pdf/1412.7753.pdf">Mikolov, T., Joulin, A., Chopra, S., Mathieu, M., & Ranzato, M. A. (2014). Learning longer memory in recurrent neural networks. arXiv preprint arXiv:1412.7753.</a><br/>

<a href='https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037'>
Neural Network Methods for Natural Language Processing Synthesis Lectures on Human Language Technologies</a><br/>

<a href='https://colah.github.io/posts/2015-08-Understanding-LSTMs/'>Understanding LSTM Networks</a><br/>

<a href=''></a>


```python

```
