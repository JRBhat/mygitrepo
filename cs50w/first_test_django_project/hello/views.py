from django.shortcuts import render, HttpResponse

# Create your views here.
def index(request):
    return render(request, "hello/index.html")


def custom(request):
    return HttpResponse("Hello JB")


def jesus(request):
    return HttpResponse("Oh Lord, bless us!")


def greet(request, name):
    # return HttpResponse(f"Hello, {name.capitalize()}")
    return render(request, "hello/greet.html", {
        "name": name.capitalize()
    })