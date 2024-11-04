from django.urls import path
from .views import INDEX,REGISTER,ABOUTUS,FRONTVIEW
from .views import CustomPasswordResetView, CustomPasswordResetDoneView, CustomPasswordResetConfirmView, CustomPasswordResetCompleteView



urlpatterns = [
path('',INDEX.as_view(), name='index'),
path('register',REGISTER.as_view(), name='register'),
path('aboutus',ABOUTUS.as_view(), name='aboutus'),
path('frontview',FRONTVIEW.as_view(), name='front-page'),
path('password_reset/', CustomPasswordResetView.as_view(), name='password_reset'),
path('password_reset/done/', CustomPasswordResetDoneView.as_view(), name='password_reset_done'),
path('reset/<uidb64>/<token>/', CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
path('reset/done/', CustomPasswordResetCompleteView.as_view(), name='password_reset_complete'),

]