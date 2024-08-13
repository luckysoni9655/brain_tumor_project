# app.py
import streamlit as st
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('braintumor.keras')
print(model.summary())
print("Number of layers:", len(model.layers))

# Function to preprocess the uploaded image
def preprocess_image(image):
   
    # Resize image to match model input shape
    image = cv2.resize(image, (150, 150))
    # Normalize pixel values
  
   
   
    img_array=np.array(image)/255.0
  
 # Add batch dimension
    img_array = np.expand_dims(image, axis=0)
  
    return img_array


# Streamlit app
def main():
    
    page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://2.bp.blogspot.com/-sznuCyjrpy0/VzWwGdRoPzI/AAAAAAAABOg/orpW6jZRVfozNmGoCuWer5k_dCIhdnb8gCLcB/s1600/brain%2Btumor%2Bcance.jpg');
            background-size: cover;    
        }
        [data-testid="stHeader"] {
            background-color:rgba(0,0,0,0);
        </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    

    st.title('Brain Tumor Detection')

    # File uploader for image
    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
    
        # Convert prediction to tumor label
        labels = ['glioma tumor', 'meningioma tumor', 'notumor', 'pituitary tumor']
        tumor_label = labels[np.argmax(prediction)]

        # Display prediction result
        st.write(f"<p style='color:red; font-size:50px;'>{tumor_label}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


    
