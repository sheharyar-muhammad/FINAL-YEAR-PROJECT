# Generated by Django 5.0.1 on 2024-06-03 15:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0003_record_prediction'),
    ]

    operations = [
        migrations.AddField(
            model_name='record',
            name='group',
            field=models.CharField(default='Unknown', max_length=100),
        ),
    ]
