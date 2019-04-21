from django.urls import path
from . import views


urlpatterns = [
    path('', views.PaintView.as_view()),
]