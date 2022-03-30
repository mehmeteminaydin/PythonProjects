from rest_framework import serializers 
from .models import Complaint 

class ComplaintSerializers(serializers.ModelSerializer): 
    class meta: 
        model=Complaint 
        fields='__all__'