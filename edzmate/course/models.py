from django.db import models

class Course(models.Model):
    course_id = models.AutoField(primary_key=True)
    course_name = models.CharField(max_length=255)
    start_date = models.DateField()
    end_date = models.DateField()
    syllabus = models.URLField()
    course_work = models.URLField()
    textbook = models.ForeignKey('textbook.Textbook', on_delete=models.CASCADE)
    professor = models.ForeignKey('professor.Professor', on_delete=models.CASCADE)

    def __str__(self):
        return self.course_name
