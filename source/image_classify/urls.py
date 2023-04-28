from django.urls import path
from .views import ImageProcessing

urlpatterns = [
    path('process-image/', ImageProcessing.as_view(), name='process_image'),
]
