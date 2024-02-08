# Import the libraries
import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.signal import wiener
from scipy.fft import fft2, ifft2


# Set the tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create a title for the app
st.title("OCR Web App with OpenCV and Pytesseract")

# Create a file uploader widget to upload the image file
uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Display the original color image
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Process the image and do all possible pre-processing steps
    sharp_amount = st.slider("Sharpness", 0, 10, 3) # Create a slider for sharpness

    # Create a kernel for sharpening
    kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)

    # Display the sharpened image
    st.image(sharpened, caption="Sharpened Image")

    
    # Process the image and do all possible pre-processing steps
    #sharp_amount = st.slider("Sharpness", 0, 10, 3,step=2) # Create a slider for sharpness
    #st.image(sharp_amount, caption="Sharp")
     # Color (RGB) Blur
    b, g, r = cv2.split(img)
    b_blur = cv2.GaussianBlur(b, (0, 0), sharp_amount)
    g_blur = cv2.GaussianBlur(g, (0, 0), sharp_amount)
    r_blur = cv2.GaussianBlur(r, (0, 0), sharp_amount)
    color_blur = cv2.merge([b_blur, g_blur, r_blur])
    

    #edge detected
    edge_amount = st.slider("Edge Detection", 0, 10, 3) # Create a slider for edge detection
    edge_detected = cv2.Canny(color_blur, edge_amount, edge_amount * 5) # Detect edges
    st.image(edge_detected, caption="Edge Detect")
    
    #blur
    blur_amount = st.slider("Blur", 1, 50, 5,step=2) # Create a slider for blur with odd values only
    blur = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0) # Blur the image
    st.image(blur, caption="Blur")
    
    #de-blur
    # Define the psf
    psf = np.ones((5, 5), np.float32) / 25.0

    # Convolve the img and the psf in the spatial domain
    convolved = cv2.filter2D(img, -1, psf)

    # Perform Wiener deconvolution in the frequency domain
    convolved_freq = fft2(convolved)
    psf_freq = fft2(psf)
    deblur_freq = wiener(convolved_freq, psf_freq,noise=0.1)

    # Inverse Fourier transform to get the deblurred image
    deblur = np.abs(ifft2(deblur_freq))

    # Display the deblurred image
    st.image(deblur, caption="De-blur")


    ''' psf = np.ones((5, 5), np.float32) / 25.0  # Example: A simple average filter as PSF
    ratio = 0.1
    # Perform Wiener deconvolution in the frequency domain
    img_freq = fft2(img)
    psf_freq = fft2(psf, s=img.shape)
    deblur_freq = wiener(img_freq, psf_freq, ratio=ratio)

    # Inverse Fourier transform to get the deblurred image
    deblur = np.abs(ifft2(deblur_freq))

    # Display the deblurred image
    st.image(deblur, caption="De-blur")'''

    '''deblur_amount = st.slider("De-blur", 0,100, 4,step=2) # Create a slider for de-blur

    psf = np.ones((5, 5), np.float32) / 25.0  # Example: A simple average filter as PSF

    # Deblur the image using Wiener deconvolution
    deblur = cv2.deconvolution(img, psf, ratio=deblur_amount)

    # Display the deblurred image
    st.image(deblur, caption="De-blur")'''
    # If image contains text do OCR and print text here
    extracted_text = pytesseract.image_to_string(img) # Extract text from the image

    if extracted_text:
        st.write("Extracted Text: ", extracted_text.strip()) # Display the text if any
    else:
        st.write("No text found.") # Display a message if no text found


'''# Check if the file is uploaded
if uploaded_file is not None:
    # Read the uploaded file as an image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Process the image and do all possible pre-processing steps
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    sharp_amount = st.slider("Sharpness", 0, 10, 3) # Create a slider for sharpness
    #nor_blur = cv2.blur(ksize=(8,3))
    #nor_blur_amt = st.slider("Normal Blur",0,10,5)
    sharp = cv2.GaussianBlur(gray, (0,0), sharp_amount) # Sharpen the image
    st.image(sharp, caption="Sharp")
    edge_amount = st.slider("Edge Detection", 0, 200, 30) # Create a slider for edge detection
    edge_detected = cv2.Canny(sharp, edge_amount, edge_amount * 5) # Detect edges
    st.image(edge_detected, caption="Edge Detect")
    blur_amount = st.slider("Blur", 0, 10, 5) # Create a slider for blur
    blur = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0) # Blur the image
    st.image(blur, caption="Blur")
    deblur_amount = st.slider("De-blur", 0, 50, 10) # Create a slider for de-blur
    deblur = cv2.fastNlMeansDenoisingColored(img,None,deblur_amount,deblur_amount,7,21) # Deblur the image
    st.image(deblur, caption="De-blur")

    # Display all pre-processed images with titles
    #st.image(sharp, caption="Sharp")
    #st.image(nor_blur,caption="Normal Blur")
    #st.image(edge_detected, caption="Edge Detect")
    #st.image(blur, caption="Blur")
    #st.image(deblur, caption="De-blur")

    # If image contains text do OCR and print text here
    extracted_text = pytesseract.image_to_string(img) # Extract text from the image

    if extracted_text:
        st.write("Extracted Text: ", extracted_text.strip()) # Display the text if any
    else:
        st.write("No text found.") # Display a message if no text found'''
