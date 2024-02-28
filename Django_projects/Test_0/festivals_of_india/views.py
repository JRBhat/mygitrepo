from django.shortcuts import render
import datetime


festivals = {
    "Makar Sankranti": (15, 1, 2024),
    "Maha Shivratri": (8, 3, 2024),
    "Holi": (25 ,3, 2024),
    "Ugadi": (9, 3, 2024)
}



# Create your views here.
        
def index(request):
    now = datetime.datetime.now()

    for fname, datetup in festivals.items():
        if datetup[0] == now.day and datetup[1] == now.month and datetup[2] == now.year:
            return render(request, "festivals_of_india/index.html", {
                "name": fname.capitalize() # context - passing paramenters
            })
        else:
            return render(request, "festivals_of_india/index.html", {
                "name": "No festival" # context - passing paramenters
            })
