import cv2
import numpy as np
import streamlit as st
from scipy.signal import fftconvolve
from scipy.signal import wiener
from scipy.fft import fft2, ifft2

# Read an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Display the original color image
    st.image(img, caption="Original Image", use_column_width=True)

    # Define the psf
    psf = np.ones((5, 5), np.float32) / 25.0

    # Convolve the img and the psf in the spatial domain
    convolved = cv2.filter2D(img, -1, psf)

    # Perform Wiener deconvolution in the frequency domain
    img_freq = fft2(img)
    psf_freq = fft2(psf, s=img.shape)
    noise = 0.5
    deblur_freq = img_freq * np.conj(psf_freq) / (np.abs(psf_freq)**2 + noise)

    # Inverse Fourier transform to get the deblurred image
    deblur_spatial = np.abs(ifft2(deblur_freq))

    # Display the deblurred image
    st.image(deblur_spatial, caption="Wiener De-blur (Frequency Domain)")
    '''img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Display the original color image
    st.image(img, caption="Original Image", use_column_width=True)

    # Define the psf
    psf = np.ones((5, 5), np.float32) / 25.0

    # Convolve the img and the psf in the spatial domain
    convolved = cv2.filter2D(img, -1, psf)

    # Perform Wiener deconvolution in the spatial domain
    deblur_spatial = wiener(convolved, mysize=(5, 5), noise=0.1)

    # Display the deblurred image
    st.image(deblur_spatial, caption="Wiener De-blur (Spatial Domain)")'''
 