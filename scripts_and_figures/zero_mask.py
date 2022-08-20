import galsim
import numpy as np
import matplotlib.pyplot as plt

im_size = 128
pixel_scale = 0.2
num_centers = 10000

psfs = [
    galsim.Gaussian(fwhm=0.6).shear(g1=0.02, g2=0.03),
    galsim.Gaussian(fwhm=0.7).shear(g1=0.03, g2=-0.03),
    galsim.Gaussian(fwhm=0.8).shear(g1=-0.01, g2=0.04)
]

images = [galsim.Image(im_size, im_size, scale=pixel_scale) for i in range(len(psfs))]
for im in images: im.setCenter(0,0)

weights = [galsim.ImageI(im_size, im_size, scale=pixel_scale, init_value=1) for i in range(len(psfs))]
for wt in weights: wt.setCenter(0,0)

# weight 0 is all 1 (no mask)

# weight 1 has a 5x5 square of 0s at the center
weights[1][galsim.BoundsI(-2,2,-2,2)].setZero()

# weight2 has 5 bad_columns scattered between -20 and 20
bad_cols = np.array([-15, -11, 4, 10, 18])
print('bad_cols = ',bad_cols)
weights[2].array[:,bad_cols + im_size//2] = 0

#weights[0].write('wt0.fits')
#weights[1].write('wt1.fits')
#weights[2].write('wt2.fits')

gal = galsim.Exponential(half_light_radius=1.1).shear(g1=0.1, g2=0.05)

def compute_bias(center, gal, psfs, images, weights):

    psf_images = []
    for psf, im in zip(psfs, images):
        galsim.Convolve(psf, gal).drawImage(image=im, center=center)
        psf_im = psf.drawImage(image=im.copy(), center=center)
        psf_images.append(psf_im)

    #images[0].write('im0.fits')
    #images[1].write('im1.fits')
    #images[2].write('im2.fits')
    #psf_images[0].write('psf0.fits')
    #psf_images[1].write('psf1.fits')
    #psf_images[2].write('psf2.fits')

    coadd = images[0].copy()
    coadd.array[:,:] = np.sum([im.array * wt.array for im,wt in zip(images,weights)], axis=0)
    coadd.array[:,:] /= np.sum([wt.array for wt in weights], axis=0)
    #coadd.write('coadd.fits')

    psf_coadd = images[0].copy()
    psf_coadd.array[:,:] = np.sum([im.array * wt.array for im,wt in zip(psf_images,weights)], axis=0)
    psf_coadd.array[:,:] /= np.sum([wt.array for wt in weights], axis=0)
    #psf_coadd.write('psf_coadd.fits')

    nowt_coadd = images[0].copy()
    nowt_coadd.array[:,:] = np.sum([im.array for im in images], axis=0)
    nowt_coadd.array[:,:] /= len(images)
    #nowt_coadd.write('nowt_coadd.fits')

    nowt_psf_coadd = images[0].copy()
    nowt_psf_coadd.array[:,:] = np.sum([im.array for im in psf_images], axis=0)
    nowt_psf_coadd.array[:,:] /= len(images)
    #nowt_psf_coadd.write('nowt_psf_coadd.fits')

    # HSM works best if the galaxy is ~centered on a postage stamp, so cut out something symmetric
    # around the galaxy.
    b = galsim.BoundsI(int(center.x)-30, int(center.x)+31,
                       int(center.y)-30, int(center.y)+31)
    hsm = galsim.hsm.EstimateShear(coadd[b], psf_coadd[b])
    nowt_hsm = galsim.hsm.EstimateShear(nowt_coadd[b], nowt_psf_coadd[b])

    #print('nowt shear = ',nowt_hsm.corrected_e1, nowt_hsm.corrected_e2)
    #print('with wt shear = ',hsm.corrected_e1, hsm.corrected_e2)
    bias_e1 = hsm.corrected_e1 - nowt_hsm.corrected_e1
    bias_e2 = hsm.corrected_e2 - nowt_hsm.corrected_e2
    return bias_e1, bias_e2

rng = np.random.default_rng(1234)

x = []
y = []
b1 = []
b2 = []
for n in range(num_centers):
    center = galsim.PositionD(*rng.uniform(-30,30,size=2))
    print(f'{n}/{num_centers}: {center}')

    try:
        bias = compute_bias(center, gal, psfs, images, weights)
    except RuntimeError:
        # Should be rare, but it happens.
        # When it does, just skip this iteration.
        print('  hsm failed')

    x.append(center.x)
    y.append(center.y)
    b1.append(bias[0])
    b2.append(bias[1])

fig, ax = plt.subplots(1,2)
div_cmap=plt.get_cmap('RdYlBu')
h1 = ax[0].hexbin(x, y, b1, cmap=div_cmap, vmin=-5e-3, vmax=5e-3)
h2 = ax[1].hexbin(x, y, b2, cmap=div_cmap, vmin=-5e-3, vmax=5e-3)

for a in ax:
    a.set_xlabel('X')
    a.set_ylabel('Y')
    a.set_aspect('equal')

ax[0].set_title('E1 bias')
ax[1].set_title('E2 bias')

# A diverging colormap that is supposedly colorblind-friendly.
fig.colorbar(h2, ax=ax, shrink=0.5)
fig.savefig('mask_bias.pdf')
fig.savefig('mask_bias.png')

print('Mean bias = ',np.mean(b1),np.mean(b2))
print('+- ',np.std(b1)/np.sqrt(len(b1)), np.std(b2)/np.sqrt(len(b2)))
