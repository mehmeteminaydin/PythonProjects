from django.db import models
from django import forms
from .models import Complaint

class ComplaintForm(forms.ModelForm):
    class Meta:
              model = Complaint
              fields = "__all__"

    title = models.TextField()
    complaint = models.TextField()