from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from django import forms
class NewForm(UserCreationForm):
    password1 = forms.CharField(max_length=16, widget=forms.PasswordInput(attrs={ 'placeholder': 'Enter Password'}))
    password2 = forms.CharField(max_length=16, widget=forms.PasswordInput(attrs={ 'placeholder': 'Password confirm'}))
    class Meta(UserCreationForm):
        model = User
        fields = UserCreationForm.Meta.fields + ("email",)
        widgets ={
            "username" : forms.TextInput(
                attrs={
                    "placeholder" : "Enter UserName"
                }
            ),
            "email" : forms.TextInput(
                attrs={
                    "placeholder" : "Enter Email"
                }
            )
        }
    
            
