# Generated by Django 3.0.5 on 2020-04-22 05:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rectifi', '0004_auto_20200421_2130'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='data',
            field=models.FileField(upload_to=''),
        ),
    ]