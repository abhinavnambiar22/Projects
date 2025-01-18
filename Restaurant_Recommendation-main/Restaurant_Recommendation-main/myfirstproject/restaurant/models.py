# models.py

from django.db import models
import requests

class UserLogin(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=100)  # In practice, use a more secure way to store passwords, such as hashing

    def __str__(self):
        return self.username

class Cuisine(models.Model):
    name=models.CharField(max_length=100)
    def __str__(self) -> str:
        return self.name

class CuisineLocation(models.Model):
    cuisineID=models.ForeignKey(Cuisine,on_delete=models.CASCADE)
    locationID=models.ForeignKey('Location',on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.cuisineID

class Location(models.Model):
    name=models.CharField(max_length=100)
    address=models.TextField()
    map_link=models.CharField(max_length=500,null=True,blank=True)
    price_range_lower=models.IntegerField(blank=True,null=True)
    price_range_upper=models.IntegerField(blank=True,null=True)
    city=models.CharField(max_length=100,blank=True,null=True)
    cuisineID=models.ManyToManyField('Cuisine')
    
    def __str__(self) -> str:
        return self.name
    
class OpeningHours(models.Model):
    day=models.CharField(max_length=100)
    open_time=models.TimeField()
    close_time=models.TimeField()
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.day

class Review(models.Model):
    rating=models.FloatField()
    comment=models.TextField()
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    userLoginID=models.ForeignKey(UserLogin,on_delete=models.CASCADE,blank=True,null=True)
    def __str__(self) -> str:
        return self.comment

class Image(models.Model):
    url=models.CharField(max_length=500)
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.url

class Tag(models.Model):
    name=models.CharField(max_length=100)
    def __str__(self) -> str:
        return self.name

class TagLocation(models.Model):
    tagID=models.ForeignKey(Tag,on_delete=models.CASCADE)
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.tagID

class OpeningHoursLocation(models.Model):
    openingHoursID=models.ForeignKey(OpeningHours,on_delete=models.CASCADE)
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.openingHoursID

class CasualDining(models.Model):
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    happyhour=models.BooleanField()
    outdoor_seating=models.BooleanField()
    live_music=models.BooleanField()

class FineDining(models.Model):
    locationID=models.ForeignKey(Location,on_delete=models.CASCADE)
    dress_code=models.CharField(max_length=100)
    michelin_star=models.IntegerField()
    