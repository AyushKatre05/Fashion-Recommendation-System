import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

# Load the feature vectors and filenames
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filename = pickle.load(open("filename.pkl", "rb"))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

# Load and preprocess the image
img = cv2.imread("2693.jpg")
img = cv2.resize(img, (224, 224))
img = np.array(img)
expand_img = np.expand_dims(img, axis=0)
pre_img = preprocess_input(expand_img)
result = model.predict(pre_img).flatten()
normalized = result / norm(result)

# Create and fit the NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)

# Find nearest neighbors
distances, indices = neighbors.kneighbors([normalized])

for file in indices[0][1:6]:  # Change to [1:6] to display 5 images
    imgName = cv2.imread(filename[file])
    cv2.imshow("Frame", cv2.resize(imgName, (640, 480)))
    cv2.waitKey(0)
