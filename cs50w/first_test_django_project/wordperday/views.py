from django.shortcuts import render
from datetime import datetime



# Create your views here.

def index(request):

    
    with open(r"D:\Code\archive\mygitrepo\cs50w\first_test_django_project\wordperday\bin\wordlist.txt", "r+") as wordlist:
        word_list = wordlist.readlines()
        # load pickle dict here
        for w in word_list:
            if w not in pickle_dict:
                
                
    return 