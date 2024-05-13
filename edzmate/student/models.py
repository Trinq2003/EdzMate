from django.db import models

class Student(models.Model):
    student_id = models.AutoField(primary_key=True)
    student_name = models.CharField(max_length=255)
    contact_information = models.TextField()
    email = models.EmailField()
    phone_number = models.CharField(max_length=20)

    def __str__(self):
        return self.student_name
