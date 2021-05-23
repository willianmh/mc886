import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from math import sqrt

Kick2 = pd.read_csv('data/kick2.dat', delimiter=' ', names=['x', 'y','z'])

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

X = Kick2.x.values
Y = Kick2.y.values
Z = Kick2.z.values

t = np.linspace(0, 19, 20)

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



for i in kick_data:
    print(kick_data[i]['data'])



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
        alpha = 1.0 / (700 + m * (epoch + 1))

      theta = theta - alpha * gradients

    #calculate the cost function
    cost_func = (loss ** 2)/2
    cost.append(cost_func)

  return theta, cost


import time

n_epochs = 1000
tol = 0.001

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


t = np.linspace(0, 49, 50)
t = np.c_[np.ones((t.shape[0], 1)), t]

X_line = t.dot(kick_data['X']['theta']['SGD'])
Y_line = t.dot(kick_data['Y']['theta']['SGD'])
Z_line = t.dot(kick_data['Z']['theta']['SGD'])



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
