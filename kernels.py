# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:21:29 2023

@author: oleja
"""

import GPy
import matplotlib.pyplot as plt
import numpy as np
#%%%

np.random.seed(12310)

X = np.arange(-25,25,0.1).reshape(-1,1)
k1 = GPy.kern.RatQuad(1,lengthscale=2,power=6, active_dims=[0])
k2 = GPy.kern.RatQuad(1,lengthscale=2,power=3,active_dims=[0])
zeros = np.zeros(X.shape[0])
ksum = k1+k2
kmult = k1 * k2
linear = k1.K(X,X)
cosine = k2.K(X,X)


comp = k1.K(linear,linear)

fig = plt.figure(layout="constrained",figsize=(12, 8))

#samples from the linear kernel
C_lin = k1.K(X,X)
linears = np.random.multivariate_normal(zeros,C_lin,3).T
ax1 = fig.add_subplot(321)
ax1.plot(X,linears,linewidth=3)
ax1.tick_params(labelbottom=False,labelsize=18)
ax1.grid()

#sample from cosine kernel
C_cos = k2.K(X,X)
periods = np.random.multivariate_normal(zeros,C_cos,3).T
ax2 = fig.add_subplot(322)
ax2.plot(X,periods,linewidth=3)
ax2.tick_params(labelbottom=False,labelsize=18)
ax2.grid()

#linear + cosine
C_sum = ksum.K(X,X)
sums = np.random.multivariate_normal(zeros,C_sum,3).T
ax3 = fig.add_subplot(323)
ax3.plot(X,sums,linewidth=3)
ax3.tick_params(labelbottom=False,labelsize=18)
ax3.grid()

#linear * cosine
C_mult = kmult.K(X,X)
mults = np.random.multivariate_normal(zeros,C_mult,3).T
ax4 = fig.add_subplot(324)
ax4.plot(X,mults,linewidth=3)
ax4.tick_params(labelbottom=False,labelsize=18)
ax4.grid()

#linear o cosine
C_comp1 = k1.K(cosine,cosine)
comp1 = np.random.multivariate_normal(zeros,C_comp1,3).T
ax5 = fig.add_subplot(325)
ax5.plot(X,comp1,linewidth=3)
ax5.tick_params(labelsize=18)
ax5.grid()

#cosine o linear
C_comp2 = k2.K(linear,linear)
comp2 = np.random.multivariate_normal(zeros,C_comp2,3).T
ax6 = fig.add_subplot(326)
ax6.plot(X,comp2,linewidth=3)
ax6.tick_params(labelsize=18)
ax6.grid()


#deal with bounding boxes
lin = fig.text(0.27, 0.75, 'LIN', ha='center', size = 22, backgroundcolor='white')
lin.set_bbox({'facecolor':'white', 'edgecolor':'red'})

cos = fig.text(0.77, 0.75, 'COS', ha='center', size = 22, backgroundcolor='white')
cos.set_bbox({'facecolor':'white', 'edgecolor':'red'})

sum_plot = fig.text(0.27, 0.425, 'LIN + COS', ha='center', size = 22, backgroundcolor='white')
sum_plot.set_bbox({'facecolor':'white', 'edgecolor':'red'})

mult_plot = fig.text(0.77, 0.425, 'LIN x COS', ha='center', size = 22, backgroundcolor='white')
mult_plot.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

comp1_plt = fig.text(0.27, 0.1, 'LIN $\circ$ COS', ha='center', size = 22, backgroundcolor='white')
comp1_plt.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

comp2_plt = fig.text(0.77, 0.1, 'COS $\circ$ LIN', ha='center', size = 22, backgroundcolor='white')
comp2_plt.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

fig.text(0.5, -0.05, 'X values, -', ha='center', size = 18)

#plt.savefig('kernel_combinations_smaller.png', dpi=500)

plt.show()
