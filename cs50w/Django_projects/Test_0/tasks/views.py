from django.shortcuts import render
from django import forms
from django.http import HttpResponseRedirect
from django.urls import reverse


# tasks = ["foo", "bar", "baz"] ##Bad way to store tasks - everybody sees this task variable - need private task list -- use sessions

class NewTaskForm(forms.Form): # django forms 
    task = forms.CharField(label="New Task")
    priority = forms.IntegerField(label="Priority", min_value=1, max_value=5)


#IF using sessions, make sure to run --> python manage.py migrate to avoid the table error

# Create your views here.
def index(request):
    if "tasks" not in request.session: # session is a big dict with all the data within that session of user
        request.session["tasks"] = [] 
    return render(request, "tasks/index.html", {
        "tasks": request.session["tasks"] 
        })


def add(request):
    if request.method == "POST":
        form = NewTaskForm(request.POST)
        if form.is_valid():
            task = form.cleaned_data["task"]
            request.session["tasks"] += [task]

            return HttpResponseRedirect(reverse("tasks:index")) # redirecting 
        else:
            return render(request, "tasks/add.html", {
                "form": form
            })
    return render(request, "tasks/add.html", {
        "form": NewTaskForm()
    })