#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np


class ANN():
    
    def __init__(self, nn_structure, eta):
        self.nn_structure = nn_structure
        self.eta = eta
        
        
    def set_weights(self,weights,b):
        self.W = weights
        self.b = b

    def f(self,x):
      return 1 / (1 + np.exp(-x))

    def f_deriv(self,x):
      return self.f(x) * (1 - self.f(x))

    def feed_forward(self,x):
      h = {1:x}
      z = {}
      for l in range(1,len(self.W)+1):
        if l == 1:
          node_in = x
        else:
          node_in = h[l]
        z[l+1] = self.W[l].dot(node_in) + self.b[l]
        h[l+1] =  self.f(z[l+1])
      return h, z

    def init_tri_values(self):
      tri_W = {}
      tri_b = {}
      for l in range(1,len(self.nn_structure)):
        tri_W[l] = np.zeros((self.nn_structure[l], self.nn_structure[l-1]))
        tri_b[l] = np.zeros((self.nn_structure[l],))
      return tri_W, tri_b


    def calculate_out_layer_delta(self,y,h_out,z_out):
      return -(y-h_out)*self.f_deriv(z_out)


    def calculate_hidden_delta(self,delta_plus_1,w_l,z_l):
      return np.dot(np.transpose(w_l),delta_plus_1) * self.f_deriv(z_l)


    def fit(self, X, y, iter_num= 10000):  
      print("Entrenando..")
      cnt = 0
      m = len(y)
      avg_cost_func = []
      while cnt < iter_num:
        if cnt%1000 == 0:
          print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = self.init_tri_values()
        avg_cost = 0
        for i in range(len(y)):
          delta = {}
          h, z = self.feed_forward(X[i,:])
          for l in range(len(self.nn_structure), 0, -1):
            if l == len(self.nn_structure):
              delta[l] = self.calculate_out_layer_delta(y[i, :], h[l], z[l])
              avg_cost += np.linalg.norm((y[i,:]-h[l]))
            else:
              if l > 1:
                delta[l] = self.calculate_hidden_delta(delta[l+1], self.W[l], z[l])
              tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
              tri_b[l] += delta[l+1]
        for l in range(len(self.nn_structure) - 1, 0, -1):
          self.W[l] += -self.eta * (1.0/m * tri_W[l])
          b[l] += -self.eta * (1.0/m * tri_b[l])
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
      return self.W, self.b


    def predict(self, X):
      m = X.shape[0]
      y = np.zeros(X.shape)
      for i in range(m):
        h, z = self.feed_forward(X[i, :])
        y[i] = h[len(self.nn_structure)]
      print(y)
      return y


# In[17]:


w1 = np.array([[.15, .2], [.25, .3]])
w2 = np.array([[.4, .45], [.5, .55]])
b1 = np.array([.35, .35])
b2 = np.array([.6, .6])

W = {1:w1, 2:w2}
b = {1:b1, 2:b2}

X = np.array([[.05, .1]])
y = np.array([[.01, .99]])


# In[18]:


nn_structure = [2,2,2]
model = ANN(nn_structure, .5)
model.set_weights(W,b)
y_pred = model.predict(X)
model.fit(X, y)
y_pred2 = model.predict(X)

