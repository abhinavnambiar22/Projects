# views.py

from django.shortcuts import render, redirect,get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from .models import UserLogin, Location, Review, Cuisine, CuisineLocation
from django.db.models import Avg
from django.db.models import Count,Subquery,OuterRef,Min
from django.db.models import QuerySet
from django.contrib.auth.hashers import check_password
from django.contrib.auth.models import User
import json

def create_superuser(username, password, email=None):
    # Check if a user with the provided username already exists
    if not User.objects.filter(username=username).exists():
        # Create a new superuser
        User.objects.create_superuser(username=username, email=email, password=password)
        print("Superuser created successfully.")
    else:
        print("Superuser creation failed. Username already exists.")

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirmed_password = request.POST['confirm_password']  # New line
        
        # Check if password and confirmed password match
        if password != confirmed_password:
            messages.error(request, 'Password and confirmed password do not match.')
            return redirect('register')

        # Check if username or email already exists
        if UserLogin.objects.filter(username=username).exists() or UserLogin.objects.filter(email=email).exists():
            messages.error(request, 'Username or email already exists.')
            return redirect('register')

        # Create new user
        user = UserLogin.objects.create(username=username, email=email, password=password)
        create_superuser(username,password,email=email)
        messages.success(request, 'Registration successful. Please login.')
        return redirect('login')

    return render(request, 'restaurant/register.html', {'foo': 'bar'})

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        try:
            # Retrieve the user from the UserLogin model based on the provided username
            user = UserLogin.objects.get(username=username)
            
            # Check if the provided password matches the password stored in the database
            if password==user.password:
                # If the password matches, authenticate the user and login
                authenticated_user = authenticate(request, username=username, password=password)
                if authenticated_user is not None:
                    login(request, authenticated_user)
                    return redirect('home')  # Redirect to the home page
                else:
                    # Handle invalid authentication
                    messages.error(request, 'Invalid username or password.')
                    return redirect('login')
            else:
                # Handle incorrect password
                messages.error(request, 'Invalid username or password.')
                return redirect('login')
        except UserLogin.DoesNotExist:
            # Handle user not found
            messages.error(request, 'Invalid username or password.')
            return redirect('login')
    
    return render(request, 'restaurant/login.html')

def index(request):
    return render(request, 'restaurant/index.html',{'user': request.user})

def detail(request, location_id):
    location = get_object_or_404(Location, pk=location_id)
    opening_hours = location.openinghours_set.all()
    img=location.image_set.all()
    reviews = location.review_set.all()
    avg_rating = reviews.aggregate(Avg('rating'))['rating__avg']
    print(location.map_link)
    context = {'location': location, 'reviews': reviews, 'avg_rating': avg_rating, 'opening_hours': opening_hours, 'img': img}
    return render(request, 'restaurant/detail.html', context)

@login_required
def add_review(request):
    if request.method == 'POST':
        # Assuming the form contains 'location_id', 'comment', and 'rating' fields
        location_id = request.POST.get('location_id')
        comment = request.POST.get('comment')
        rating = request.POST.get('rating')
        
        # Assuming you have a logged in user
        user = request.user
        
        # Create a new review object
        review = Review.objects.create(location_id=location_id, comment=comment, rating=rating, user=user)
        review.save()
        
        # Redirect to the same page or any other page
        return redirect('home')  # Redirect to home page after adding review
        
    # If not a POST request, just redirect back to home page
    return redirect('home')


@login_required
def logout_view(request):
    logout(request)
    return redirect(reverse('login'))

def restaurant(request):
    
    raw_sql = 'SELECT "restaurant_location"."id", "restaurant_location"."name", "restaurant_location"."address", "restaurant_location"."map_link", AVG("restaurant_review"."rating") AS "avg_rating" FROM "restaurant_location" LEFT OUTER JOIN "restaurant_review" ON ("restaurant_location"."id" = "restaurant_review"."locationID_id") GROUP BY "restaurant_location"."id", "restaurant_location"."name", "restaurant_location"."address", "restaurant_location"."map_link" ORDER BY "restaurant_location"."name" ASC'

    locations = Location.objects.raw(raw_sql)
    
  
    # Fetching cuisine of a specific location
    context = {'locations': locations}
    print(locations)
    return render(request, 'restaurant/restaurant.html', context)

def recommendation(request):
    # Fetching all locations
    
    raw_sql = 'SELECT "restaurant_location"."id", "restaurant_location"."name", "restaurant_location"."address", "restaurant_location"."map_link", AVG("restaurant_review"."rating") AS "avg_rating" FROM "restaurant_location" LEFT OUTER JOIN "restaurant_review" ON ("restaurant_location"."id" = "restaurant_review"."locationID_id") GROUP BY "restaurant_location"."id", "restaurant_location"."name", "restaurant_location"."address", "restaurant_location"."map_link" ORDER BY "restaurant_location"."name" ASC'

    locations = Location.objects.raw(raw_sql)
    locations_with_avg_rating = Location.objects.annotate(avg_rating=Avg('review__rating'))
    
    # Group locations by cuisine
    cuisine_restaurants = {}
    for location in locations_with_avg_rating:
        cuisine_name = location.cuisineID.name
        if cuisine_name not in cuisine_restaurants or location.avg_rating > cuisine_restaurants[cuisine_name].avg_rating:
            cuisine_restaurants[cuisine_name] = location
    '''SELECT c.Name AS Cuisine, r.Name AS Restaurant, AVG(rt.Rating) AS AverageRating
FROM Restaurant r
JOIN Cuisine c ON r.CuisineID = c.ID
LEFT JOIN Rating rt ON r.ID = rt.RestaurantID
GROUP BY c.Name, r.Name
HAVING AVG(rt.Rating) = (
    SELECT MAX(AverageRating)
    FROM (
        SELECT AVG(rt.Rating) AS AverageRating
        FROM Restaurant r
        JOIN Cuisine c ON r.CuisineID = c.ID
        LEFT JOIN Rating rt ON r.ID = rt.RestaurantID
        GROUP BY c.Name, r.Name
    ) AS temp
    WHERE temp.Cuisine = c.Name
);
'''
    cuisines = Cuisine.objects.annotate(num_restaurants=Count('location')).order_by('-num_restaurants')
    '''SELECT c.Name, COUNT(r.ID) AS NumRestaurants
FROM Cuisine c
JOIN Restaurant r ON c.ID = r.CuisineID
GROUP BY c.Name
ORDER BY NumRestaurants DESC;'''


    locations_higher_than_avg_in_city = Location.objects.annotate(
        avg_rating=Subquery(
            Review.objects.filter(locationID_id=OuterRef('id')).values('locationID_id').annotate(avg_rating=Avg('rating')).values('avg_rating')
        )
    ).filter(avg_rating__gt=Subquery(
        Review.objects.filter(locationID_id__in=Location.objects.filter(city='Manipal').values('id')).values('locationID_id').annotate(avg_rating=Avg('rating')).values('avg_rating')
    )).values('name')

    '''SELECT name
    FROM Location
    WHERE (
        SELECT AVG(rating)
        FROM Review
        WHERE locationID_id = Location.id
    ) > (
        SELECT AVG(rating)
        FROM Review
        WHERE locationID_id IN (
            SELECT id
            FROM Location
            WHERE city = 'Manipal'
        )
    );
    '''
    

    locations_with_lower_price_range_less_than_cuisine_avg = Location.objects.annotate(
        min_price_range_lower=Min('price_range_lower')
    ).filter(
        min_price_range_lower__lt=Subquery(
            Location.objects.filter(cuisineID__in=CuisineLocation.objects.filter(locationID=OuterRef('id')).values('cuisineID')).values('price_range_lower').annotate(avg_price_range_lower=Avg('price_range_lower')).values('avg_price_range_lower')
        )
    ).values('name')

    locations_casual = Location.objects.filter(casualdining__outdoor_seating=True).exclude(casualdining__live_music=True)

    '''SELECT * 
    FROM myapp_location 
    INNER JOIN myapp_casualdining 
    ON myapp_location.id = myapp_casualdining.locationID_id 
    WHERE myapp_casualdining.outdoor_seating = True 
    AND NOT EXISTS 
        (SELECT 1 
        FROM myapp_casualdining AS cd 
        WHERE cd.locationID_id = myapp_location.id 
        AND cd.live_music = True);'''
    
    locations_fine = Location.objects.filter(finedining__michelin_star__gt=2)

    '''SELECT * 
        FROM myapp_location 
        INNER JOIN myapp_finedining 
        ON myapp_location.id = myapp_finedining.locationID_id 
        WHERE myapp_finedining.michelin_star > 2;'''

    # Fetching cuisine of a specific location
    
    # Dynamically generate filter options based on the data
    # Initialize empty lists for each option
    filter_options = {
    'Outdoor seating but no live music for Casual Dining': sorted(set(locations_casual.values_list('name', flat=True))),
    'Most common cuisines': sorted(set(cuisines.values_list('name', flat=True))),
    'Rating Greater than avg in City': sorted(set(locations_higher_than_avg_in_city.values_list('name', flat=True))),
    'Fine Dining with More than 2 Michellin Star': sorted(set(locations_fine.values_list('name', flat=True))),
    }

    # Add "All" option to each list
    filter_options_json = json.dumps(filter_options)
    print(filter_options_json)
    
    context = {'restaurants': cuisine_restaurants.values(), 'cuisines': cuisines,'locations': locations, 'locations_higher_than_avg_in_city': locations_higher_than_avg_in_city, 'locations_with_lower_price_range_less_than_cuisine_avg': locations_with_lower_price_range_less_than_cuisine_avg,
               'filter_options_json': filter_options_json, 'filter_options': filter_options}
    
    return render(request, 'restaurant/recommendation.html', context)

def about(request):
    return render(request, 'restaurant/about.html',{'foo': 'bar'})

