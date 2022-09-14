"""
This script shows the result of using inverse variance weighting including both signal and
background noise for a linear coadd.

It makes Figure 1 of the final version of the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# color-blind friendly colors for line plots from https://gist.github.com/thriveth/8560036
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CB_color_names = ['blue', 'orange', 'green',
                  'pink', 'brown', 'purple',
                  'grey', 'red', 'yellow']
colors = dict(zip(CB_color_names, CB_color_cycle))

mean=0
sig_1 = 0.5
sig_2 = 1.5
sig_3 = 3.0
b_1 = 10
b_2 = 10
b_3 = 10

min_x = -10
max_x = 10
dx = 0.01
x_values = np.arange(min_x, max_x, dx)

star_flux_vals = [1, 1.e5]

im_1_color = colors['blue']
im_2_color = colors['orange']
im_3_color = colors['green']
star_color_vals = ['black', colors['pink']]

object_sig = 0.0

fig = plt.figure()
ax = fig.add_subplot(111)
printed = False
star_ind = 0
for flux_val in star_flux_vals:
    print('Beginning %f...'%flux_val)
    total_sig_1 = np.sqrt(object_sig**2 + sig_1**2)
    total_sig_2 = np.sqrt(object_sig**2 + sig_2**2)
    total_sig_3 = np.sqrt(object_sig**2 + sig_3**2)
    im_1 = flux_val * scipy.stats.norm(mean, total_sig_1).pdf(x_values)
    im_2 = flux_val * scipy.stats.norm(mean, total_sig_2).pdf(x_values)
    im_3 = flux_val * scipy.stats.norm(mean, total_sig_3).pdf(x_values)
    w_1 = 1./(b_1 + im_1)
    w_2 = 1./(b_2 + im_2)
    w_3 = 1./(b_3 + im_3)
    tot_im = (im_1*w_1 + im_2*w_2 + im_3*w_3)/(w_1+w_2+w_3)
    if not printed:
        ax.plot(x_values, im_1/(im_1.sum()*dx), label=r'$I_1(x)$', color=im_1_color, linewidth=0.8)
        ax.plot(x_values, im_2/(im_2.sum()*dx), label=r'$I_2(x)$', color=im_2_color, linewidth=0.8)
        ax.plot(x_values, im_3/(im_3.sum()*dx), label=r'$I_3(x)$', color=im_3_color, linewidth=0.8)
        printed = True
    ax.plot(x_values, tot_im/(tot_im.sum()*dx), label=r'$I_{coadd}(x)$, flux=$10^%d$'%np.log10(flux_val),
            color=star_color_vals[star_ind], linewidth=2.0)
    print(np.median(tot_im*im_2.sum()/(tot_im.sum()*im_2)))
    star_ind += 1
ax.set(xlabel='x', ylabel=r'$I(x)$/flux')
ax.set_yscale('log')
ax.set_ylim([1e-5,1])
plt.legend(loc='upper right')

plt.savefig('coadd_psf.png')
plt.savefig('coadd_psf.pdf')
