from django.urls import path
from .views import UserInputView, ResultView

urlpatterns = [
    path('input/', UserInputView.as_view(), name='input'),
    path('result', ResultView.as_view(), name='result'),
]
