# musicplayer/views.py
from django.shortcuts import render, redirect
from .models import Song
from .forms import SongForm

def home(request):
    if request.method == 'POST':
        form = SongForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = SongForm()

    songs = Song.objects.all()
    return render(request, 'musicplayer/home.html', {'form': form, 'songs': songs})
