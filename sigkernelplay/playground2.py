#%%
import matplotlib.pyplot as plt
import numpy as np
import sigkernel
import torch
#t = np.arange(0, 10).reshape(-1, 1)
#s = np.arange(0,20).reshape(-1, 1)


#t=torch.from_numpy(t).unsqueeze(0).double()
#s=torch.from_numpy(s).unsqueeze(0).double()


static_kernel = sigkernel.LinearKernel()
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)


#print(signature_kernel.compute_kernel(t, s))


t = np.arange(0, 20)
s = np.arange(0, 10)
x=t
y=np.cos(s) #s
def efm(X, lam, t_grid):
    T=np.max(t_grid)
    dX=np.diff(X)
    t_grid_p1=t_grid[1:]
    t_grid=t_grid[:-1]
    coeff=(np.exp(-lam*(T - t_grid_p1))-np.exp(-lam*(T - t_grid)))/(lam*(t_grid_p1 - t_grid ))
    dX1=dX*coeff
    X1=np.cumsum(dX1)
    return X1


lam=np.array([0.5])
xefm = efm(x, lam, t).reshape(-1,1)
yefm = efm(y, lam, s).reshape(-1,1)
plt.plot(xefm[:,0])
plt.plot(yefm[:,0])
plt.show()

xefm = torch.from_numpy(xefm).unsqueeze(0).double()
yefm = torch.from_numpy(yefm).unsqueeze(0).double()


print(signature_kernel.compute_kernel(xefm, yefm))
print(xefm.shape)

#%%





