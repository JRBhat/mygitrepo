from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, "hello/index.html") # to render an entire html file #NOTE: Django best pracice(use appname/filename.html) for namespace considerations

def jb(request):
    return HttpResponse("Hello Jb!") # just to pass an http response

def kkk(request):
    return HttpResponse("kkk ok!")

# def greet(request, name):
#     return HttpResponse(f"Hello {name.capitalize()}!") # passing parameters

def greet_better(request, name):
    return render(request, "hello/greet.html", {
        "name": name.capitalize() # context - passing paramenters
    })

