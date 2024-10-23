# -*- coding: utf-8 -*-
"""DEL_lab3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10mcb9NNvrPJgZnuQhCDUc1a-HKmeUajH
"""

from numpy import *
import pandas as pd

def tan_h(x):
  return tanh(x)

def threshold(x):
  if x > 0:
    return 1
  else:
    return 0

# 1
inputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]

def check():
  w = matrix([0.5, 0.5])
  for i in inputs:
    i = matrix(i)
    h = (w*i.T)[0,0] # hidden layer
    t =  tan_h(h) # hidden layer activation

    y = threshold(t) # output layer activation
    print("Input:",i," Output:", y)

check()

# 2
def softmax(v):
  e = exp(v)
  return e/sum(e)

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data)
w1 = array([[0.5]*4]*3)
w2 = array([[0.1]*3]*3)

# hidden layer
h1 = matrix(df)*w1.T
y1 = tanh(h1) # hidden layer output

# output layer
h2 = matrix(y1)* w2.T
#y2 = softmax(h2)

#print(h2)

l = []
for i in h2:
  l.append(softmax(i))
display(l)