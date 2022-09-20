This directory contains scripts that were used to create the figures, along with the figures themselves:

Figure 1 is produced using `psf_tests_inverse_variance.py`, and the resulting figure is written to `coadd_psf.png` and `coadd_psf.pdf`.

Figure 2 is produced using `psf_tests_median.py`, and the resulting figure is written to `noisy_median_coadd_psf.png` and `noisy_median_coadd_psf.pdf`.

Figure 3 is produced using `median_bias.py` (which also relies on utilities in `coadd_schemes.py`) and the resulting figure is in `median_coadd_bias_dist.png` and `median_coadd_bias_dist.pdf`.

Figure 4 is produced using `extended.py`, and the resulting figure is written to `extended.pdf` and `extended.png`.


The calculations in Sec 3.2, leading up to Eq. 16 (TBC) were numerically confirmed with this notebook:
https://colab.research.google.com/drive/1_mO1ly5Lj3B0gCzYPcQqnhP_frjooXvz?usp=sharing
