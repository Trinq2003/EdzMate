from django.db import models

class Lesson(models.Model):
    lesson_id = models.AutoField(primary_key=True)
    assignment = models.URLField()
    lesson_note = models.URLField()
    exercise = models.URLField()
    instruction = models.URLField()
    solution = models.URLField()
    course = models.ForeignKey('course.Course', on_delete=models.CASCADE)

    def __str__(self):
        return f'Lesson {self.lesson_id} for {self.course}'
