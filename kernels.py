# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:21:29 2023

@author: oleja
"""



import GPy
import matplotlib.pyplot as plt
import numpy as np
#%%%

#Plotting samples form the MVN

np.random.seed(12310)

X = np.arange(0,50,0.1).reshape(-1,1)


ls1=10
ls2=1
var1=1
var2=1
par1 = "-"
par2 = "-"

#Here play around with kernel types
k1 = GPy.kern.RBF(1,variance=var1, lengthscale=ls1, active_dims=[0])
k2 = GPy.kern.RBF(1,variance=var2, lengthscale=ls2,active_dims=[0])


ktype1 = str(k1._name).upper()
ktype2 = str(k2._name).upper()

if (len(ktype1) > 3) or (len(ktype2) > 3):
    ktype1 = ktype1[:3]
    ktype2 = ktype2[:3]


if ktype1 == ktype2:
    ktype1 = ktype1 + "1"
    ktype2 = ktype2 + "2"





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

#plt.imshow(C_lin)

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
lin = fig.text(0.415, 0.72, f'{ktype1}\nL={ls1},$\sigma^2$={var1}', ha='center', size = 18, backgroundcolor='white')
lin.set_bbox({'facecolor':'white', 'edgecolor':'red'})

cos = fig.text(0.91, 0.72, f'{ktype2}\nL={ls1},$\sigma^2$={var2}', ha='center', size = 18, backgroundcolor='white')
cos.set_bbox({'facecolor':'white', 'edgecolor':'red'})

sum_plot = fig.text(0.4, 0.4, f'{ktype1} + {ktype2}', ha='center', size = 18, backgroundcolor='white')
sum_plot.set_bbox({'facecolor':'white', 'edgecolor':'red'})

mult_plot = fig.text(0.91, 0.4, f'{ktype1} x {ktype2}', ha='center', size = 18, backgroundcolor='white')
mult_plot.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

comp1_plt = fig.text(0.41, 0.08, f'{ktype1} $\circ$ {ktype2}', ha='center', size = 18, backgroundcolor='white')
comp1_plt.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

comp2_plt = fig.text(0.91, 0.08, f'{ktype2} $\circ$ {ktype1}', ha='center', size = 18, backgroundcolor='white')
comp2_plt.set_bbox({ 'facecolor':'white', 'edgecolor':'red'})

fig.text(0.5, -0.05, 'X values, -', ha='center', size = 18)

#plt.savefig('kernel_combinations_smaller.png', dpi=500)

plt.show()


#%%%
#Plotting the covarianc ematrices


ls1=10
ls2=1
var1=1
var2=1
par1 = "-"
par2 = "-"


X = np.arange(0,50,0.1).reshape(-1,1)
k1 = GPy.kern.RBF(1,variance=var1,lengthscale=ls1, active_dims=[0])
#k1 = GPy.kern.Linear(1,10, active_dims=[0])
k2 = GPy.kern.RBF(1,variance=var2,lengthscale=ls2,active_dims=[0])


ktype1 = str(k1._name).upper()
ktype2 = str(k2._name).upper()

if (len(ktype1) > 3) or (len(ktype2) > 3):
    ktype1 = ktype1[:3]
    ktype2 = ktype2[:3]


if ktype1 == ktype2:
    ktype1 = ktype1 + "1"
    ktype2 = ktype2 + "2"
    



ksum = k1+k2
kmult = k1 * k2
linear = k1.K(X,X)
cosine = k2.K(X,X)
comp = k1.K(linear,linear)

fig = plt.figure(figsize=(16, 8))

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=-0.15,
                    hspace=0.2)




kernels_used = [f"{ktype1}: L={ls1}, $\sigma^2$={var1}, a = {par1}", f"{ktype2}: L={ls2}, $\sigma^2$ ={var2}, a = {par2}",\
                f"{ktype1} + {ktype2}", f"{ktype1} x {ktype2}", f"{ktype1}$\circ${ktype2}", f"{ktype2}$\circ${ktype1}"]

#samples from the linear kernel
C_lin = k1.K(X,X)

ax1 = fig.add_subplot(231)
im = ax1.imshow(C_lin)
#ax1.tick_params(labelbottom=False,labelsize=18)
#ax1.grid()
ax1.set_title(kernels_used[0], fontsize=18)
ax1.tick_params(labelbottom=False,labelsize=15)
ax1.set_ylim(500,0)
ax1.set_xlim(0,500)
#fig.colorbar(im1,ax=ax1)




#sample from cosine kernel
C_k2 = k2.K(X,X)

ax2 = fig.add_subplot(232)
im = ax2.imshow(C_k2)
ax2.set_title(kernels_used[1],fontsize=18)
ax2.tick_params(labelbottom=False, labelleft=False)
#ax2.grid()
#fig.colorbar(im2,ax=ax2)



#linear + cosine
C_sum = ksum.K(X,X)

ax3 = fig.add_subplot(233)
im = ax3.imshow(C_sum)
ax3.grid()
ax3.set_title(kernels_used[2],fontsize=18)
ax3.tick_params(labelbottom=False, labelleft=False)
#fig.colorbar(im3,ax=ax3)
ax3.grid()


#linear * cosine
C_mult = kmult.K(X,X)
ax4 = fig.add_subplot(234)
im = ax4.imshow(C_mult)
ax4.set_title(kernels_used[3],fontsize=18)
ax4.tick_params(labelsize=15)
ax4.set_xlim(0,500)
ax4.set_ylim(500,0)
ax4.set_xticks([0,100,200,300,400,500])
#ax4.grid()


#linear o cosine
C_comp1 = k1.K(cosine,cosine)

ax5 = fig.add_subplot(235)
im= ax5.imshow(C_comp1)
ax5.tick_params(labelleft=False,labelsize=15)
ax5.set_title(kernels_used[4],fontsize=18)
ax5.set_xlim(0,500)
ax5.set_ylim(500,0)
ax5.set_xticks([0,100,200,300,400,500])

ax5.grid()

#cosine o linear
C_comp2 = k2.K(linear,linear)

ax6 = fig.add_subplot(236)
im = ax6.imshow(C_comp2)
ax6.tick_params(labelleft=False,labelsize=15)
fig.colorbar(im,ax=[ax1,ax2,ax4,ax5,ax3,ax6])
fig.axes[6].tick_params(axis='both', labelsize=18)

ax6.set_title(kernels_used[5],fontsize=18)
ax6.set_xlim(0,500)
ax6.set_ylim(500,0)
ax6.set_xticks([0,100,200,300,400,500])
ax6.grid()



