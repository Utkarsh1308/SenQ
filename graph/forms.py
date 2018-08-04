from django.contrib.auth.models import User
from django import forms

class ContactForm(forms.Form):
    alpha = forms.CharField(label='alpha', max_length=100)

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
