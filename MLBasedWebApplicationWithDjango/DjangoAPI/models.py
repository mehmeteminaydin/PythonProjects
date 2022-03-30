from django.db import models


class Complaint(models.Model ):
    title = models.TextField()
    complaint = models.TextField()

    def __str__(self):
        return self.title

class FromDataBase(models.Model):
    baslik = models.TextField()
    metin = models.TextField()
    tarih = models.DateField()
    kullanici = models.TextField()
    nitelikler = models.TextField()

    def __str__(self):
        return self.kullanici
