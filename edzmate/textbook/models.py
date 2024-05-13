from django.db import models

class Textbook(models.Model):
    textbook_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    table_of_content = models.URLField()  # URL to XML file

    def __str__(self):
        return self.title
