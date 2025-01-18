# Generated by Django 5.0.1 on 2024-04-18 19:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('restaurant', '0009_location_price_range_casualdining_finedining'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='location',
            name='price_range',
        ),
        migrations.AddField(
            model_name='location',
            name='price_range_lower',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='location',
            name='price_range_upper',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
