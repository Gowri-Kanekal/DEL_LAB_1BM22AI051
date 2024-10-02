# -*- coding: utf-8 -*-
"""DEL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1En7xHL2JxO91n5TrL_sDqw8pIU57mjDY

Demonstrate the following activation functions:
1. Threshold function
2. Sigmoid
3. Relu
4. Leaky Relu
5. SoftMax
6. tanh

**Notes**

*   Threshold fuction is a binary step function, with discrete values 0 or 1 only.
*   sigmoid - [0,1] best suited for bianry classification, output layers -- cons: vanishing gradient


*   tanh - [-1,1] used in hidden layers
*   ReLU

How do you address multi-calss problems?  -- use softmax activation function
"""

# 1. Threshold function

def threshold(x):
  if x > 0:
    return 1
  else:
    return 0

a = int(input("enter a number: "))
threshold(a)

x = linspace(-10,10,100)
y =[]
for i in x:
  y.append(threshold(i))
plt.plot(x,y)
plt.grid()
plt.show()

# 2. Sigmoid

from numpy import *

def sigmoid(x):
  return 1/(1+exp(-x))

a = float(input("enter a number: "))
sigmoid(a)

import matplotlib.pyplot as plt
def plot_graph():
  x = linspace(-10,10,100)
  y = sigmoid(x)
  plt.plot(x,y)
  plt.grid()
  plt.show()
plot_graph()

# 3. relu

def relu(x):
  if x > 0:
    return x
  else:
    return 0

a = int(input("enter a number: "))
relu(a)

import matplotlib.pyplot as plt

x = linspace(-10,10,100)
y =[]
for i in x:
  y.append(relu(i))
plt.plot(x,y)
plt.grid()
plt.show()

# 4. leaky relu

def leaky_relu(x):
  if x>0:
    return x
  else:
    return 0.01*x

a = int(input("enter a number: "))
leaky_relu(a)

x = linspace(-10,10,100)
y =[]
for i in x:
  y.append(leaky_relu(i))
plt.plot(x,y)
plt.grid()
plt.show()

# 5. tanh

def tan_h(x):
  return tanh(x)

a = int(input("enter a number: "))
tan_h(a)

import matplotlib.pyplot as plt
def plot_graph():
  x = linspace(-10,10,100)
  y = tan_h(x)
  plt.plot(x,y)
  plt.grid()
  plt.show()
plot_graph()

# implement a simple neural network to predict XOR values

a = int(input("enter a number: "))
b = int(input("enter a number: "))

w = matrix([0.5, -0.5])
i = matrix([a, b])
h = (w*i.T)[0,0]
t =  tan_h(h)
print(int(bool(t)))