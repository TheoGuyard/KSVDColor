import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data, util
from sklearn.feature_extraction import image
from ksvdcolor.ksvd import KSVD

# --- Parameters --- #

img_file = "img/4.1.05.tiff"

# Image and noise params
sigma = 0.1                     # noise standard deviation
patch_size = (8, 8)             # patch dimensions
n = np.prod(patch_size)         # number of pixels per patch
C = 1.15                        # noise gain

# KSVD params
k = 32                          # number of atoms
maxiter = 5                     # number of KSVD iterations
omp_tol = n * (sigma * C)**2    # OMP tolerance defined in [1]
omp_nnz = 30                    # OMP non-zero coeffs targeted over the n coeffs
use_sklearn = False             # whether to use sklean for the OMP step
param_a = 0.5                   # parameter for the modified scalar-product


# --- Image pre-processing --- #

# Read image and get the cmap for plotting
img = mpimg.imread(img_file)
img = util.img_as_float(img)
cmap = 'gray' if len(img.shape) == 2 else None

# Add the noise
img_with_noise = util.random_noise(img, var=sigma**2)

# Extract and concatenate patches to obtain the input
patches = image.extract_patches_2d(img, patch_size)
Y = patches.reshape(patches.shape[0], -1)


# --- Denoising --- #

ksvd = KSVD(k=k, maxiter=maxiter, omp_tol=omp_tol, omp_nnz=omp_nnz, 
            use_sklearn=use_sklearn, param_a=param_a)
ksvd.learn_dictionary(Y)
alpha = ksvd.denoise(Y)

# --- Reconstruct image and plot --- #

img_reconstructed = alpha @ ksvd.dictionary
img_reconstructed = image.reconstruct_from_patches_2d(
    img_reconstructed.reshape(patches.shape), img.shape)

plt.subplot(131)
plt.imshow(img, cmap=cmap)
plt.subplot(132)
plt.imshow(img_with_noise, cmap=cmap)
plt.subplot(133)
plt.imshow(img_reconstructed, cmap=cmap)