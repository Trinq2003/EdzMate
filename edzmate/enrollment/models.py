from django.db import models

class Enrollment(models.Model):
    enrollment_id = models.AutoField(primary_key=True)
    course = models.ForeignKey('course.Course', on_delete=models.CASCADE)
    student = models.ForeignKey('student.Student', on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.student} enrolled in {self.course}'
