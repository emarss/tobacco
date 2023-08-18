from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import os
import uuid
import os
import numpy as np
import face_recognition


class ImageProcessing(APIView):
    def post(self, request):
        image_file_1 = request.FILES.get('image_1')
        image_file_2 = request.FILES.get('image_2')
        img_1 = Image.open(image_file_1)
        img_2 = Image.open(image_file_2)
        filename_1 = str(uuid.uuid4()) + os.path.splitext(image_file_1.name)[1]
        filename_2 = str(uuid.uuid4()) + os.path.splitext(image_file_2.name)[1]
        processed_img_1_path = os.path.join('/tmp/', filename_1)
        processed_img_2_path = os.path.join('/tmp/', filename_2)
        img_1.save(processed_img_1_path)
        img_2.save(processed_img_2_path)

        image1 = face_recognition.load_image_file(processed_img_1_path)
        image2 = face_recognition.load_image_file(processed_img_2_path)

        face_landmarks1 = face_recognition.face_landmarks(image1)
        face_landmarks2 = face_recognition.face_landmarks(image2)

        if len(face_landmarks1) == 0:
            print("First image does not contain a face.")
            exit()

        if len(face_landmarks2) == 0:
            print("Second image does not contain a face.")
            exit()

        face_encoding1 = face_recognition.face_encodings(image1)[0]
        face_encoding2 = face_recognition.face_encodings(image2)[0]

        results = face_recognition.compare_faces([face_encoding1], face_encoding2)
        if results[0]:
            return Response({'result': "MATCH"}, status=status.HTTP_200_OK)
        else:
            return Response({'result': "NOT_MATCH"}, status=status.HTTP_200_OK)