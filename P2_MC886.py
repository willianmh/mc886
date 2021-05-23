#!/usr/bin/env python
# coding: utf-8

# # Projeto 2 - Métodos de Aprendizagem Supervisionada
# 
# ### Grupo
# - Ismael Pereira Santos de Melo - RA175460
# - Willian Massahiro Hayashida - RA188705
# 

# # Loading Data

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from math import sqrt

Kick2 = pd.read_csv('data/kick2.dat', delimiter=' ', names=['x', 'y','z'])


# ## Visualizing kicks

# In[2]:


fig = plt.figure(figsize=(8, 6), dpi=80)

ax = plt.axes(projection="3d")
ax.scatter3D(Kick2.x, Kick2.y, Kick2.z, c='b');

X_field,Y_field = np.meshgrid(np.linspace(-3,3,4), np.linspace(0,2,2))
Z_field = np.zeros(X_field.shape)
ax.plot_surface(X_field,Y_field,Z_field,shade=False, color='g', alpha=0.4)
ax.plot([-2.5 , -2.5, 2.5, 2.5], [0, 1, 1, 0], zdir ='z', zs=0, c='g')
ax.plot([-1.25, -1.25, 1.25, 1.25], [0, 0.05, 0.05, 0], zdir ='z', zs=0, c='g')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim(-3, 3)
ax.set_ylim(0 ,2)
ax.set_zlim(0, 0.5)

plt.show()


# In[3]:


X = Kick2.x.values
Y = Kick2.y.values
Z = Kick2.z.values


# In[4]:


t = np.linspace(0, 19, 20)


# # Training Model

# In[5]:


kick_data = {
    'X' : {
        'data' : X,
        'theta' : {},
        'cost' : {}   
    },
    'Y' : {
        'data' : Y,
        'theta' : {},
        'cost' : {}   
    },
    'Z' : {
        'data' : Z,
        'theta' : {},
        'cost' : {}   
    }
}


# In[6]:


for i in kick_data:
    print(kick_data[i]['data'])


# ## Stochastic Gradient Descent

# In[7]:


def stochasticGradientDescent(X_b, y, lr='opt', eta0=0.001, n_epochs=50, tol=0.0005):
  m = X_b.shape[0] 
#   cost = np.zeros(n_epochs)
  cost = []
  X_b = np.c_[np.ones((X_b.shape[0], 1)), X_b] 
  theta = np.random.rand(X_b.shape[1])
    
  X_b, y = shuffle(X_b, y)
  cost_func = 100000
  for epoch in range(n_epochs):
    if cost_func < tol:
      break
     
    for i in range(m):
      xi = X_b[i: i+1]
      yi = y[i: i+1]
      
      # calculate its gradient
      loss = xi[0].dot(theta) - yi
      gradients = xi[0].T * loss[0]

      alpha = eta0
      # update parameters
      if(lr == 'opt'):
        alpha = 1.0 / (780 + m * (epoch + 1))
        
      theta = theta - alpha * gradients
      
    #calculate the cost function
    cost_func = (loss ** 2)/2
    cost.append(cost_func)
 
  return theta, cost


# In[8]:


import time

n_epochs = 10000
tol = 0.00005

for i in kick_data:
    starter_time = time.time()
    theta, cost = stochasticGradientDescent(t, kick_data[i]['data'], lr='opt', n_epochs=n_epochs, tol=tol) 
    time_1 = time.time() - starter_time
    
    num_iterations = len(cost)
    print("training: ", i)
    print("- time of training linear regression with SGD: ", time_1)
    print("- minimun cost: ", min(cost))
    print("- num of iterations: ", num_iterations)
    print()
    
    kick_data[i]['theta']['SGD'] = theta
    kick_data[i]['cost']['SGD'] = cost


# In[9]:


fig = plt.figure(figsize=(8, 8), dpi=80)

j = 1
for i in kick_data:
    num_iterations = len(kick_data[i]['cost']['SGD'])
    ax = plt.subplot(2, 2, j)
    ax.plot(np.arange(0,num_iterations), kick_data[i]['cost']['SGD'], "r.")
    ax.set_ylabel('custo')
    ax.set_xlabel('numero de iterações')
    ax.set_title(i)
    j = j + 1

plt.show()


# In[10]:


def predictModel(X, Y, theta):
  X = np.c_[np.ones((X.shape[0], 1)), X] 

  predict_values = X.dot(theta)
  
  return predict_values


# In[11]:


t_line = np.linspace(0, 49, 50)

X_line = predictModel(t_line, X, kick_data['X']['theta']['SGD'])
Y_line = predictModel(t_line, Y, kick_data['Y']['theta']['SGD'])
Z_line = predictModel(t_line, Z, kick_data['Z']['theta']['SGD'])


# In[12]:


fig = plt.figure(figsize=(8, 6), dpi=80)

ax = plt.axes(projection="3d")


ax.plot3D(X_line, Y_line, Z_line, 'gray')
ax.scatter3D(Kick2.x, Kick2.y, Kick2.z, c='b');

X_field,Y_field = np.meshgrid(np.linspace(-3,3,4), np.linspace(0,2,2))
Z_field = np.zeros(X_field.shape)
ax.plot_surface(X_field,Y_field,Z_field,shade=False, color='g', alpha=0.4)
ax.plot([-2.5 , -2.5, 2.5, 2.5], [0, 1, 1, 0], zdir ='z', zs=0, c='g')
ax.plot([-1.25, -1.25, 1.25, 1.25], [0, 0.05, 0.05, 0], zdir ='z', zs=0, c='g')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim(-3, 3)
ax.set_ylim(0 ,2)
ax.set_zlim(0, 0.5)

plt.show()


# In[ ]:





# ## Gradient Descent

# In[14]:


def gradientDescent(X_b, y, lr='opt', eta0=0.005, n_epochs=50, tol=0.01):
    m = X_b.shape[0] 
    cost = np.zeros(n_epochs)
    X_b = np.c_[np.ones((X_b.shape[0], 1)), X_b] 
    theta = np.random.rand(X_b.shape[1])

    X_b, y = shuffle(X_b, y)
    cost_func = 100000
    for epoch in range(n_epochs):
        if cost_func < tol:
            break
    
        # calculate its gradient
        loss = X_b.dot(theta) - y
        gradients = X_b.T.dot(loss) / m
        
        
        alpha = eta0
      # update parameters
        if(lr == 'opt'):
            alpha = 1.0 / (250 + m * (epoch/10 + 1))
        
        theta = theta - alpha * gradients
        
      
        #calculate the cost function
        cost_func = (np.sum(loss) ** 2)/2
        cost[epoch] = cost_func
 
    return theta, cost


# In[19]:


import time

n_epochs = 1000
tol = 0.0005

for i in kick_data:
    starter_time = time.time()
    theta, cost = gradientDescent(t, kick_data[i]['data'], lr='opt', n_epochs=n_epochs, tol=tol) 
    time_1 = time.time() - starter_time
    
    num_iterations = len(cost)
    print("training: ", i)
    print("- time of training linear regression with GD: ", time_1)
    print("- minimun cost: ", min(cost))
    print("- num of iterations: ", num_iterations)
    print()
    
    kick_data[i]['theta']['GD'] = theta
    kick_data[i]['cost']['GD'] = cost


# In[20]:


fig = plt.figure(figsize=(8, 8), dpi=80)

j = 1
for i in kick_data:
    num_iterations = len(kick_data[i]['cost']['GD'])
    ax = plt.subplot(2, 2, j)
    ax.plot(np.arange(0,num_iterations), kick_data[i]['cost']['GD'], "r.")
    ax.set_ylabel('custo')
    ax.set_xlabel('numero de iterações')
    ax.set_title(i)
    j = j + 1

plt.show()


# In[22]:


t_line = np.linspace(0, 49, 50)

X_line_GD = predictModel(t_line, X, kick_data['X']['theta']['GD'])
Y_line_GD = predictModel(t_line, Y, kick_data['Y']['theta']['GD'])
Z_line_GD = predictModel(t_line, Z, kick_data['Z']['theta']['GD'])


# In[23]:


fig = plt.figure(figsize=(8, 6), dpi=80)

ax = plt.axes(projection="3d")


ax.plot3D(X_line_GD, Y_line_GD, Z_line_GD, 'gray')
ax.scatter3D(Kick2.x, Kick2.y, Kick2.z, c='b');

X_field,Y_field = np.meshgrid(np.linspace(-3,3,4), np.linspace(0,2,2))
Z_field = np.zeros(X_field.shape)
ax.plot_surface(X_field,Y_field,Z_field,shade=False, color='g', alpha=0.4)
ax.plot([-2.5 , -2.5, 2.5, 2.5], [0, 1, 1, 0], zdir ='z', zs=0, c='g')
ax.plot([-1.25, -1.25, 1.25, 1.25], [0, 0.05, 0.05, 0], zdir ='z', zs=0, c='g')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim(-3, 3)
ax.set_ylim(0 ,2)
ax.set_zlim(0, 0.5)

plt.show()


# In[ ]:



def meanSquaredError(Y_predicted, Y):
  error = 0
  m = len(Y_predicted)
  for i in range(m):
    error += ((Y_predicted[i] - Y[i])**2)
  return error/m


# In[ ]:


prices_validation = predictModel(X_validation, y_validation, theta_best_SGD_1)

from sklearn.metrics import mean_squared_error

mse_val = meanSquaredError(prices_validation, y_validation)
mse_val_sklearn = mean_squared_error(y_validation, prices_validation)

print("Error:")
print(mse_val)
print(mse_val_sklearn)

