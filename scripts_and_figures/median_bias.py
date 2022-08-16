'''
Code to produce Figure 3, "Median Coadd Bias", referenced in section 4.1 on
the Median Coadd.
Created by Michael Murphy, advised by Rachel Mandelbaum and Tianqing Zhang.

Primary file, holding __main__, to be ran from terminal.
'''
import matplotlib.pyplot as plt
import numpy as np
import galsim
from coadd_schemes import median_fitting, median_img


def create_fit_dist(low_flux, high_flux, n_noise, fwhm_dist, rng, bright=None):
    '''
    Creates galsim objects to simulate imaging, coadds the images, and finds
        the best fit based on a bright instance

    Inputs:
    low_flux: float, faint star flux count
    high_flux: float, bright star flux count
    n_noise: float, the number of noisy realizations to use
    fwhm_dist: list of float PSF sizes
    rng: random number generator (galsim.BaseDeviate object)
    bright: a galsim.Image object, which is used to fit faint realizations, if
        not given, it will be made

    Returns:
    fstar_arr: list of floats, representing the distribution of f_star best fit
        parameters, len(fstar_arr) == n_noise
    summed[0]
    '''
    #--------------------------------------------------------------------------
    # This section holds parameters static across all callings of this function
    g1 = .1                                     # shear
    g2 = .2                                     # shear
    pixel_scale = 0.2                           # arcsec / pixel
    bf = 1.e2                                   # background flux counts, float
    #--------------------------------------------------------------------------
    # Getting terminal input and setting up the necessary variables
    num_measurements = len(fwhm_dist)
    gaussian_fwhms = fwhm_dist
    flux1 = high_flux                           # bright star flux count, float
    flux2 = low_flux                            # faint star flux count, float

    bound_val = 100
    max_bound = galsim.BoundsI(1, bound_val, 1, bound_val)
            
    if isinstance(bf, float) or isinstance(bf, int):
        bf = [bf]*num_measurements
    
    # Must place brighter image first
    fluxes = [flux1, flux2]
    # Holds bright image, whether given or created
    bright_img = None
    fstar_arr = None
    #--------------------------------------------------------------------------

    if bright == None:
        fluxes_to_use = fluxes
    else:
        fluxes_to_use = [flux2]
        # The bright image must still have its flux given as high_flux
        bright_img = bright

    for flux in fluxes_to_use:
        # Creating test galaxy objects
        objs = []
        for fwhm in gaussian_fwhms:
            objs.append(galsim.Gaussian(fwhm=fwhm,
                                        flux=flux).shear(g1=g1,g2=g2))
        # Creating test galaxy images
        imgs = []
        for obj in objs:
            imgs.append(obj.drawImage( scale=pixel_scale, bounds=max_bound ))
        # Summing the images via median weighting
        # If we need to create a bright image (only once per visualize_bias)
        if not bright_img:
            bright_img = median_img(imgs, bf, n_noise, pixel_scale, rng)
        else:                                     
            fstar_arr = median_fitting(imgs, bright_img, fluxes[0],
                                       bf, n_noise, pixel_scale, rng)
        #----------------------------------------------------------------------
        # If we have our f_star distribution and are done in this function
        if fstar_arr is not None:
            return fstar_arr, bright_img
    

def visualize_bias(n_noise, rng, ax, fwhm_dist, name, color_it):
    '''
    Given an axis, this plots the bias from median coadd for the given
        fwhm_dist

    Inputs:
    n_noise: float, the number of noisy realizations to use
    rng: random number generator (galsim.BaseDeviate object)
    ax: matplotlib axis object
    fwhm_dist: list of float PSF sizes
    name: string, what to put in legend for this fwhm_dist
    color_it: int, to pick a new color
    '''
    print("Starting simulation for {} distribution".format(fwhm_dist))
    # Colorblind friendly palette
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']
    color = CB_color_cycle[color_it]

    # Creating input fluxes: the bright object to use as a model is the largest
    #   flux in this list
    flux_arr = np.geomspace(1.e1, 1.e5, num=20, endpoint=True)
    datapoints = np.empty_like(flux_arr, dtype=np.ndarray)
    # Getting bright image (and first data point)
    data, bright_img = create_fit_dist(flux_arr[0],
                                       flux_arr[-1],
                                       n_noise, fwhm_dist, rng)
    datapoints[0] = data 
    # Getting rest of the data, using the bright_img object as model for all
    for iter, val in enumerate(flux_arr[1:]):
        datapoints[iter+1] = create_fit_dist(val,
                                             flux_arr[-1],
                                             n_noise, fwhm_dist, rng,
                                             bright=bright_img)[0]
    # Turning f_star distributions into plottable data
    f_vals = np.empty_like(flux_arr)
    errors = np.empty_like(flux_arr)
    for iter, val in enumerate(datapoints):
        f_vals[iter] = (np.mean(val))
        # This is standard deviation of the mean, rather than the spread in the distribution.
        # This is the right thing to use if we want to know how well we've determined the mean bias.
        errors[iter] = (np.std(val)/np.sqrt(n_noise))

    # x-axis is ratio of input flux to bright flux
    x_vals = flux_arr / flux_arr[-1]

    #--------------------------------------------------------------------------
    lower_bound = (f_vals-errors)/flux_arr-1
    true_val = f_vals/flux_arr-1
    upper_bound = (f_vals+errors)/flux_arr-1

    ax.fill_between(x_vals, lower_bound, upper_bound, alpha=0.3, color=color)
    
    ax.plot(x_vals, true_val, color=color, linestyle= "-", \
            label = str(name))



if __name__ == "__main__":
    rng = galsim.BaseDeviate(8241573+6)
    # function to create a list with max/min determined by mult parameter
    make_list = lambda x, mult: [x,round(x*(1+mult)/2,3),round(mult*x,3)]
    # factors to use
    smallest_size = 0.6
    multiplier = [1.2,2.0,3.0]

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    # Getting Data
    n_noise = 1000
    for iter, mult in enumerate(multiplier):
        input_fwhms = make_list(smallest_size,mult)
        visualize_bias(n_noise, rng, ax,
                       input_fwhms,
                       r"$*${}: {}".format(mult,input_fwhms),
                       iter)
    ax.legend(title="PSF FWHM distribution",loc='upper right')
    ax.set(title="Bias in PSF photometry for different PSF size distributions",
           xlabel=r"$f_{*}^{~true}/f_{*}^{~bright}$",
           ylabel=r"$f_{*}^{~estimated}/f_{*}^{~true}-1$")
    # Marking where zero is
    ax.axhline(y=0, color='gray', linestyle=':')
    # Symlog Scaling
    # Marking the linear region
    linthresh = 0.01
    ax.set_yscale('symlog',linthresh=linthresh)
    ax.set_xscale('log')
    ax.axhline(y=linthresh, color='lightgray', linestyle=':')
    ax.axhline(y=-linthresh, color='lightgray', linestyle=':')
    plt.savefig('median_coadd_bias_dist.png')
    plt.savefig('median_coadd_bias_dist.pdf')
    plt.show()
