# Generated by Django 3.0.5 on 2020-04-22 05:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rectifi', '0005_auto_20200421_2220'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='data',
            field=models.ImageField(upload_to=''),
        ),
    ]