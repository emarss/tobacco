from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import os
import uuid
import cv2
import sys
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.feature import hog
import numpy as np
from joblib import load



class ImageProcessing(APIView):
    def post(self, request):
        # Get the uploaded image from the request
        image_file = request.FILES.get('image')
        # return Response('image_file');

        # Open the image using PIL
        img = Image.open(image_file)

        # Process the image (e.g. resize, crop, filter, etc.)

        # Generate a unique filename for the processed image
        filename = str(uuid.uuid4()) + os.path.splitext(image_file.name)[1]


        # Save the processed image to a path
        processed_img_path = os.path.join('/home/carter/Projects/test/kuda_api/', filename)
        img.save(processed_img_path)

        # Create a JSON response with the processed image path
        response_data = {
            'result': processed_img_path
        }

        target_size = (128, 128)

        # new_image_path = sys.argv[1]
        new_image = preprocess_image(processed_img_path, target_size)
        new_features = extract_features([new_image])

        # Load the saved model from a file
        model_file = "./tobacco_leaf_classifier.joblib"

        clf = load(model_file)

        # Use the trained model to predict the grade of the new image
        new_grade = clf.predict(new_features)
        print(new_grade)


        # Return the JSON response
        return Response(response_data, status=status.HTTP_200_OK)


def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def extract_features(X):
    features = []
    for image in X:
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm="L2-Hys")
        features.append(hog_features)
    return features