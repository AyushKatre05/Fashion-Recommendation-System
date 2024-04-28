import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Load the feature vectors and filenames
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filename.pkl', 'rb'))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('ShopSnapster -revolutionizing the way you shop at shopping-malls')

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)  # Create 'uploads' directory if it doesn't exist
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

def extract_feature(img, model):
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # Reshape the features to match the expected dimension (e.g., 1000)
    features = features[:1000]  # Adjust this based on your feature dimension

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        # feature extract
        img_array = np.array(display_image)
        preprocessed_img = preprocess_input(img_array)
        features = extract_feature(preprocessed_img, model)
        # recommendation
        indices = recommend(features, feature_list)
        # show recommended images
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(Image.open(filenames[indices[0][1]]))
        with col2:
            st.image(Image.open(filenames[indices[0][2]]))
        with col3:
            st.image(Image.open(filenames[indices[0][3]]))
        with col4:
            st.image(Image.open(filenames[indices[0][4]]))
        with col5:
            st.image(Image.open(filenames[indices[0][5]]))
    else:
        st.error("Some error occurred while saving the file.")
