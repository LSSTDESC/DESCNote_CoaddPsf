import matplotlib.pyplot as plt
import numpy as np

sigma1 = 0.5
sigma2 = 1.5
sigma3 = 3
gal_scale = 1
gal_flux = 100
dx = 0.01

x = np.linspace(-10,10, int(20/dx)+1)
P1 = np.exp(-x**2 / (2*sigma1**2)) / (2*np.pi*sigma1**2)**0.5
P2 = np.exp(-x**2 / (2*sigma2**2)) / (2*np.pi*sigma2**2)**0.5
P3 = np.exp(-x**2 / (2*sigma3**2)) / (2*np.pi*sigma3**2)**0.5

print('P fluxes = ',P1.sum()*dx, P2.sum()*dx, P3.sum()*dx)

T = np.exp(-np.abs(x)/gal_scale) / (2*gal_scale) * gal_flux

print('T flux = ',T.sum()*dx)

I1 = np.convolve(T,P1, 'same') * dx
I2 = np.convolve(T,P2, 'same') * dx
I3 = np.convolve(T,P3, 'same') * dx

print('I fluxes = ',I1.sum()*dx, I2.sum()*dx, I3.sum()*dx)

Pc = np.median([P1,P2,P3], axis=0)
Ic = np.median([I1,I2,I3], axis=0)
IPc = np.convolve(T,Pc, 'same') * dx

print('coadd fluxes = ',Pc.sum()*dx, Ic.sum()*dx, IPc.sum()*dx)

fig, ax = plt.subplots(1,1)

ax.plot(x, I1, label='$I_1(x)$', color='blue', linewidth=0.4)
ax.plot(x, I2, label='$I_2(x)$', color='orange', linewidth=0.4)
ax.plot(x, I3, label='$I_2(x)$', color='green', linewidth=0.4)
ax.plot(x, Ic, label='$I_\mathrm{coadd}(x)$', color='magenta')
ax.plot(x, IPc, label='$T \otimes P_\mathrm{coadd}(x)$', color='black')

ax.legend(loc='upper right')
ax.set_yscale('log')
ax.set_xlim(-4.5,4.5)
ax.set_ylim(0.5,70)
ax.set_xlabel('$x$')
ax.set_ylabel('$I(x)$/flux')

fig.savefig('extended.pdf')
