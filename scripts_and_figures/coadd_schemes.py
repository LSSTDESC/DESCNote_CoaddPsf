'''
Code to produce Figure 3, "Median Coadd Bias", referenced in section 4.1 on
the Median Coadd.
Created by Michael Murphy, advised by Rachel Mandelbaum and Tianqing Zhang.

Auxillary file, holding coaddition and fitting functions.
'''
import numpy as np
import galsim

#------------------------------------------------------------------------------

def find_model_flux(imaged_arr, model_arr, stddev_arr=None):
    '''
    Function to find the f_star value that minimizes χ^2

    Inputs:
    imaged_arr: an np-array that is to be fitted to
    model_arr: an np-array that is used as a model, note that it MUST be
        normalized to have flux = 1
    Note: The Standard Deviation is an optional parameter, because if the
          standard deviation is constant throughout the image, the term plays
          no role so there is no need to pass a dummy variable

    Returns:
    f_star: a float, the best-fit parameter

    Calculation of the maximum likelihood estimate of f_star:
    Given: I_star (x) = f_star * P(x), ∫P(x)dx=1 
    χ^2 = Σ(I_star (x) - f_star * P(x)/ σ )^2 (everything but f_star is indexed)
    ∂(χ^2)/∂(f_star) = ∑ 2 (I_star (x) - f_star * P(x)/ σ^2 ) (-P(x)) -> 0
    Minimum value of χ^2 at f_star = ∑ ((I_star * P)/σ^2) / ∑ (P^2/σ^2)
    '''
    if( stddev_arr is not None ):
        f_star = (np.sum(np.divide(np.multiply(imaged_arr, model_arr),np.square(stddev_arr))) /
                  np.sum(np.divide(np.square(model_arr),np.square(stddev_arr))))
    else:
        f_star = ((np.sum(np.multiply(imaged_arr, model_arr))) /
                  np.sum(np.square(model_arr)))
    return f_star

#------------------------------------------------------------------------------

def median_fitting(images, bright_image, bright_flux, bf, n_noise, pixel_scale, rng):
    '''
    Finds the median coadd f_star distribution for a set of faint images
        given a bright model to compare to.

    Inputs:
    images: a list of galsim.Image objects to add noise to and coadd
    bright_image: a galsim.Image object to be used as a model
    bright_flux: float, the true flux of bright_image
    bf: a list of int/float values, len(bf)==len(faint_images), corresponding
        to a constant background level for each image
    n_noise: an int/float value, the number of noisy realizations
    pixel_scale: an int/float value, for galsim noise object
    rng: a galsim.BaseDeviate random number generator, for creating noise
    low_flux_val: a list of int/float values 

    Returns:
    f_star_array: a 1d np-array that holds the values of our fitting process
        (to be analyzed later)
    '''
    num_imgs = len(images)
    
    # Creating a list of Noise Objects, each corresponding to a specific bf,
    #   to be applied to their corresponding images for each noise iteration
    noise_objs = [None] * (num_imgs)
    for i in range(num_imgs):
        noise_objs[i] = galsim.GaussianNoise(rng, 
                                             sigma=np.sqrt(bf[i] * pixel_scale**2))

    f_star_array = np.empty(n_noise)
    for noise_it in range(int(n_noise)):
        # Making a copy of the images to manipulate this iteration,
        #   leaving the originals untouched.
        imgs = []
        for img in images:
            imgs.append(img.copy())
        # Adding noise via the created noise objects from above
        for img, noise_obj in zip(imgs, noise_objs):
            img.addNoise( noise_obj )
        # Actually creating an array of median values:
        #   first getting the arrays
        #   then, stacking them on top of each other
        #   lastly, taking the median value across that new axis to the array
        #   is of the original dimensions
        med_array = np.median(np.stack([img.array for img in imgs], axis=2), axis=2)
        # Fitting this realization (not passing a stddev_arr as we only use background
        #   flux as uncertainty, and the background is defined as a constant)
        f_star = find_model_flux( med_array, bright_image.array/bright_flux )
        f_star_array[noise_it] = f_star
    return f_star_array

def median_img(images, bf, n_noise, pixel_scale, rng):
    '''
    Only creates an image, does not attempt fitting (used to create instance
        of bright image)
    Look to comments of median_fitting() for description of inputs.

    Returns:
    median_image: a galsim.Image object of our image inputs, after applying noise to each
        realization and taking the average of the n_noise realizations
    '''
    num_imgs = len(images)
    #Creating a list of Noise Objects, each corresponding to a specific bf,
    #   to be applied to their corresponding images for each noise iteration
    noise_objs = [None] * (num_imgs)
    for i in range(num_imgs):
        noise_objs[i] = galsim.GaussianNoise(rng,
                                             sigma=np.sqrt(bf[i] * pixel_scale**2))

    total_array = np.zeros_like(images[0].array)
    for noise_it in range(int(n_noise)):
        # Making a copy of the images to manipulate this iteration, leaving the
        #   originals untouched
        imgs = []
        for img in images:
            imgs.append(img.copy())
        # Adding noise via the created noise objects from above
        for img, noise_obj in zip(imgs, noise_objs):
            img.addNoise( noise_obj )
        # Actually creating an array of median values:
        #   first actually getting the arrays
        #   then, stacking them on top of each other
        #   lastly, taking the median value across that new axis to the array
        #   is of the original dimensions
        med_array = np.median( np.stack([img.array for img in imgs], axis=2), axis=2)
        total_array += med_array
    # We take the mean of the median values of all the realizations to get our
    #   final image
    total_array /= n_noise
    median_image = galsim.Image( total_array )
    
    return median_image

