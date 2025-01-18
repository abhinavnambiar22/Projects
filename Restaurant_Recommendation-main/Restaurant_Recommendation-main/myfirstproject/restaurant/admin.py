from django.contrib import admin
from .models import Location,UserLogin, Cuisine, OpeningHours, Review, Image, Tag, TagLocation, CuisineLocation, OpeningHoursLocation, CasualDining,FineDining

# Register your models here.
admin.site.register(Location)
admin.site.register(UserLogin)
admin.site.register(Cuisine)
admin.site.register(OpeningHours)
admin.site.register(Review)
admin.site.register(Image)
admin.site.register(Tag)
admin.site.register(TagLocation)
admin.site.register(CuisineLocation)
admin.site.register(OpeningHoursLocation)
admin.site.register(CasualDining)
admin.site.register(FineDining)
