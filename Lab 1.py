#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Lab 1
#07/10/24

#Create and implement a basic neuron model with a computational framework, integrate essential elements like input nodes, weights, bias and activation function. 
#Evaluate and observe how each activation function influences the network's behaviour and effectiveness in handling different types of data. 

#Implement a XOR function


# In[3]:


def threshold(x):
  if x > 0:
    return 1
  else:
    return 0

def sigmoid(x):
  return 1/(1+exp(-x))

def relu(x):
  if x > 0:
    return x
  else:
    return 0

def leaky_relu(x):
  if x>0:
    return x
  else:
    return 0.01*x

def tan_h(x):
  return tanh(x)


# In[4]:


def accuracy(output, yp):
  c = 0
  for i in range(len(output)):
    if output[i] == yp[i]:
      c+=1
  return (c/len(output))*100


# In[6]:


from numpy import*
inputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]

output = [0,1,1,0] # actual target values

def check_xor(function):
  yp=[]
  print("\n", function)
  for i in inputs:
    w = matrix([0.5, -0.5])
    i = matrix(i)
    h = (w*i.T)[0,0]
    t =  function(h)
    print("Input:",i," Output:", int(bool(t)))
    yp.append(int(bool(t))) # predicted target values

  print("Accuracy:",accuracy(output, yp),"%")

check_xor(threshold)
check_xor(sigmoid)
check_xor(relu)
check_xor(leaky_relu)
check_xor(tan_h)

