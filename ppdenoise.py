#  Reference:
#  Peter Kovesi, "Phase Preserving Denoising of Images". 
#  The Australian Pattern Recognition Society Conference: DICTA'99. 
#  December 1999. Perth WA. pp 212-217
#  https://www.peterkovesi.com/papers/denoise.pdf

#  Original work(Matlab) : Copyright (c) 1998-2000 Peter Kovesi
#  Python implementation : Copyright (c) 2021 Lafith Mattara
#  www.peterkovesi.com
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in 
#  all copies or substantial portions of the Software.
# 
#  The Software is provided "as is", without warranty of any kind.

#  September 1998 - original version
#  May 1999       - 
#  May 2000       - modified to allow arbitrary size images
#  June 2021      - Python implementation

import numpy as np


def ppdenoise(
        img,
        k=2, nscale=5,
        mult=2.5, norient=6,
        softness=1.0):
    print(type(img))
    """Function to denoise an image while preserving phase information

    Parameters
    ----------
    img : numpy.ndarray
        Original noisy image as numpy array
    k : int, optional
         No of standard deviations of noise to reject 2-3,
         by default 2
    nscale : int, optional
        No of filter scales to use (5-7) - the more scales used
        the more low frequencies are covered
        , by default 5
    mult : float, optional
        multiplying factor between scales  (2.5-3),
        by default 2.5
    norient : int, optional
        No of orientations to use (6),
        by default 6
    softness : float, optional
        degree of soft thresholding (0-hard  1-soft),
        by default 1.0

    Returns
    -------
    numpy.ndarray
        Denoised image
    """
    min_wavelength = 2
    sigma_onf = 0.55
    dtheta_onsigma = 1.0
    epsilon = 0.00001

    theta_sigma = np.pi/norient/dtheta_onsigma

    img_fft = np.fft.fft2(img)
    row, col = img_fft.shape

    x = np.matmul(
            np.ones((row, 1)),
            (np.arange(
                -col/2, col/2)/(col/2)).reshape(1, -1))
    y = np.matmul(
            np.arange(-row/2, row/2).reshape(-1, 1),
            np.ones((1, col))/(row/2)
            )

    radius = np.sqrt(np.square(x)+np.square(y))
    radius[int(row/2), int(col/2)] = 1

    theta = np.arctan2(-y, x)
    total_energy = np.zeros((row, col))

    estmean_en = []
    sig = []

    for orient in np.arange(1, norient+1):
        print("Processing orientation {}".format(orient))

        angl = (orient - 1)*np.pi/norient
        wavelength = min_wavelength

        ds = np.subtract(
                np.sin(theta)*np.cos(angl),
                np.cos(theta)*np.sin(angl))
        dc = np.add(
                np.cos(theta)*np.cos(angl),
                np.sin(theta)*np.sin(angl))
        dtheta = np.abs(np.arctan2(ds, dc))

        spread = np.exp(-np.square(dtheta)/(2*theta_sigma**2))

        for scale in np.arange(1, nscale+1):
            # print("Scale : ",scale)
            f_o = 1.0/wavelength
            rf_o = f_o/0.5

            log_gabor = np.exp(
                    -np.square(
                        np.log(
                            radius/rf_o))/(2*np.log(sigma_onf)**2)
                        )
            log_gabor[int(row/2), int(col/2)] = 0

            filter = log_gabor*spread
            filter = np.fft.fftshift(filter)

            e0_fft = img_fft*filter
            e0 = np.fft.ifft2(e0_fft)
            ae0 = np.abs(e0)

            if scale == 1:
                median_en = np.median(ae0.reshape(1, row*col))
                mean_en = (0.5*np.sqrt(-np.pi/np.log(0.5))) * median_en

                ray_var = (4-np.pi)*np.square(mean_en)/np.pi
                ray_mean = mean_en

                estmean_en.append(mean_en)
                sig.append(np.sqrt(ray_var))

            t = (ray_mean + k*np.sqrt(ray_var))/(np.power(mult, scale-1))
            valid_e0 = (ae0 > t)

            v = np.divide(softness*t*e0, (ae0+epsilon))
            v = np.add(
                    np.multiply(
                        np.invert(valid_e0).astype(int),
                        e0),
                    np.multiply(
                        valid_e0.astype(int),
                        v)
                    )

            e0 = e0-v
            total_energy = total_energy + e0
            wavelength = wavelength*mult

    clean_image = np.real(total_energy)
    return clean_image
