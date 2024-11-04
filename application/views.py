from django.shortcuts import render
from django.views.generic import TemplateView, CreateView
from .form import NewForm
from django .urls import reverse_lazy
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView



# Create your views here.

class INDEX(TemplateView):
    template_name = 'index.html'

class REGISTER(CreateView):
    form_class = NewForm
    template_name = 'Auth/register.html'
    success_url = reverse_lazy('index')

class ABOUTUS(TemplateView):
    template_name = 'aboutus.html'

class FRONTVIEW(TemplateView):
    template_name = 'front_page.html'  

class CustomPasswordResetView(PasswordResetView):
    template_name = 'Auth/password_reset_form.html'
    success_url = '/password_reset/done/'

class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'Auth/password_reset_done.html'

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'Auth/password_reset_confirm.html'
    success_url = '/reset/done/'

class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'Auth/password_reset_complete.html'    



      
