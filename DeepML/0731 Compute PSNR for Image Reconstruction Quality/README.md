# [731. Compute PSNR for Image Reconstruction Quality](https://www.deep-ml.com/problems/731)

## Problem Description

Peak Signal-to-Noise Ratio (PSNR) is a widely used metric for measuring the quality of a reconstructed or compressed image compared to the original. It is commonly reported in image denoising, super-resolution, and compression tasks. Higher PSNR values generally indicate that the reconstruction is closer to the original image.

Implement a function `compute_psnr(original, reconstructed, max_pixel_value)` that computes the PSNR (in decibels) between two images of the same shape.

### Input

- `original`: a numpy array representing the original image (any shape)
- `reconstructed`: a numpy array of the same shape as `original`
- `max_pixel_value`: a float representing the maximum possible pixel value of the image (e.g., 255 for 8-bit images, 1.0 for normalized images)

### Output

- Return the PSNR value as a float. If the two images are identical, return `float('inf')`.

## Example

Input:
```
orig = np.array([[0, 0], [0, 0]], dtype=float)
recon = np.array([[1, 1], [1, 1]], dtype=float)
compute_psnr(orig, recon, 255)
```
Output:
```
48.1308
```

## Explanation

TODO

## Solution

TODO
