"""Script to illustrate features of median coadds"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import numpy.random

# Gaussian star profiles
mean=0
sig_1 = 0.5
sig_2 = 1.5
sig_3 = 3.0

# Image background levels
b_1 = 100
b_2 = 100
b_3 = 100

star_flux_vals = [1,1e5]
object_sig = 0.0

im_1_color = 'blue'
im_2_color = 'orange'
im_3_color = 'brown'
star_color_vals = ['green', 'magenta']

# Grid of x values for evaluating the image
min_x = -4
max_x = 4
dx = 0.1
x_values = np.arange(min_x, max_x, dx)
print(np.min(x_values), np.max(x_values))

# Number of noise realizations
n_noise = 50000

fig = plt.figure()
ax = fig.add_subplot(111)
printed = False
star_ind = 0
for flux_val in star_flux_vals:
    print('Beginning %f...'%flux_val)
    total_sig_1 = np.sqrt(object_sig**2 + sig_1**2)
    total_sig_2 = np.sqrt(object_sig**2 + sig_2**2)
    total_sig_3 = np.sqrt(object_sig**2 + sig_3**2)
    # Need to divide by dx because the PDF routine assumes you want the values to integrate to 1,
    # rather than summing to 1.
    im_1 = flux_val * scipy.stats.norm(mean, total_sig_1).pdf(x_values).reshape(-1,1)/dx
    im_2 = flux_val * scipy.stats.norm(mean, total_sig_2).pdf(x_values).reshape(-1,1)/dx
    im_3 = flux_val * scipy.stats.norm(mean, total_sig_3).pdf(x_values).reshape(-1,1)/dx
    # Next we need to do the coaddition some number of times depending on whether we're adding noise
    # or not.
    if n_noise == 1:
        all_ims = np.concatenate((im_1, im_2, im_3), axis=1)
        tot_im = np.median(all_ims, axis=1)
        mean_im = np.mean(all_ims, axis=1)
    else:
        all_ims = np.concatenate((im_1, im_2, im_3), axis=1)
        bg_ims = np.concatenate(
            (b_1*np.ones_like(im_1), b_2*np.ones_like(im_2), b_3*np.ones_like(im_3)), axis=1)
        tot_im = np.zeros_like(np.median(all_ims, axis=1))
        mean_im = np.zeros_like(np.median(all_ims, axis=1))
        for ival in range(n_noise):
            noisy_im = numpy.random.poisson(lam=all_ims+bg_ims) - bg_ims
            noisy_tot_im = np.median(noisy_im, axis=1)
            noisy_mean_im = np.mean(noisy_im, axis=1)
            tot_im += noisy_tot_im
            mean_im += noisy_mean_im
        tot_im /= n_noise
        mean_im /= n_noise
    im_1 = im_1[:,0]
    im_2 = im_2[:,0]
    im_3 = im_3[:,0]
    if not printed:
        ax.plot(x_values, im_1/flux_val, label=r'$I_1(x)$', color=im_1_color)
        ax.plot(x_values, im_2/flux_val, label=r'$I_2(x)$', color=im_2_color)
        ax.plot(x_values, im_3/flux_val, label=r'$I_3(x)$', color=im_3_color)
        printed = True
    if n_noise == 1:
        if star_ind == 0:
            ax.plot(x_values, mean_im/flux_val, label=r'$I_{mean}(x)$',
                    color=star_color_vals[star_ind], linestyle='--')
        ax.plot(x_values, tot_im/flux_val, label=r'$I_{median}(x)$, flux=$10^{%d}$'%np.log10(flux_val),
                color=star_color_vals[star_ind])
    else:
        if star_ind == 0:
            ax.plot(x_values, mean_im/flux_val, label=r'$\langle I_{mean}\rangle(x)$',
                    color=star_color_vals[star_ind], linestyle='--')
        ax.plot(x_values, tot_im/flux_val, label=r'$\langle I_{median}\rangle(x)$, flux=$10^{%d}$'%np.log10(flux_val),
                color=star_color_vals[star_ind])
    star_ind += 1
ax.set(xlabel='x', ylabel=r'$I(x)$/flux')
ax.set_yscale('log')
ax.set_ylim([0.1,3e1])
plt.legend()
plt.savefig('noisy_median_coadd_psf.png')
