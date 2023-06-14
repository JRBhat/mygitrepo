'''
This file acts as a one-stop place, which contains all the input data required by the application.
Data is imported from here at various points during the flow of control.
It's isolated nature helps easy modification of repeatedly used variables and constants
'''

# Assigning path containing the image files to a variable 'path'
# Python config module - suggestions; Creating config files 

import json
import subprocess
import sys
from random import randint

from Common_Functions_Stable import get_colorprofile_specs, convert_other_extensions_to_jpg, resize_jpg


input_json_path = input("Please provide the path of the Json configuration file for the study: ")

# load study config .json
with open(rf"{input_json_path}", "r") as config_file:
    data = json.load(config_file)

# extract study configuration
mypath = data['path'] 
studynumber = data['stdyno']
filename_mask = data['MASK']
path_for_validation = mypath 
No_marketing = data['No_marketing']



colorfile, colorname, file_extension = get_colorprofile_specs(path_for_validation)

if file_extension == "jpg":
    try: # resize only
        validated_path, my_list_after_resize = resize_jpg(path_for_validation)
    except TypeError:
        print("Resize limit reached. Total file size less than 400MB and is well-adjusted for PDF use")
        print("Returning original jpg path")
        validated_path = mypath

elif file_extension == "png" or file_extension == "tif": # convert and then resize
    validated_path, my_list_after_conversion_and_resize = convert_other_extensions_to_jpg(path_for_validation, file_extension)

subprocess.Popen(f"d: && start {validated_path}", shell=True)
input("Conversion complete. Please check the created images in the opened folder and then press enter")



checker = input("Are the images transformed correctly...(y/n)?")
while True:
    if checker.lower() == "y":
        file_extension = "*.jpg"
        break
    elif checker.lower() == "n": 
        print("Something wrong with the images. Stopping program..")
        sys.exit(1)
    checker = input("Are the images transformed correctly...(y/n)?")

# ................End of all Image resizing............................


Test_type = ""
randomfilepath = ""
Test_info = input("What kind of test is this(S/T/SR/TR/C): ")
Test_number = input("Iteration number: ")
if Test_number == " ":
    # seed(1) # for pseudo random numbers
    Test_number = str(randint(0,100))
if Test_info.lower() == "s":
    Test_type = f"S_{Test_number}"
    randomfilepath = None
elif Test_info.lower() == "t":
    Test_type = f"T_{Test_number}"
    randomfilepath = None
elif Test_info.lower() == "sr":
    Test_type = f"SR_{Test_number}"
    randomfilepath = data["RANDOM"]
elif Test_info.lower() == "tr":
    Test_type = f"TR_{Test_number}"
    randomfilepath = data["RANDOM"]
elif Test_info.lower() == "c":
    Test_type = f"C_{Test_number}"
    if data["RANDOM"] != "None":
        randomfilepath = data["RANDOM"]
    else:
        randomfilepath = None

draft_flag = False
# draft_flag = True

if randomfilepath == "None":
    randomfilepath = None
    Test_type = Test_type.replace("_with_Randomization", "_NoRandomfile")

if "BEIERSDORF" in data.keys():
    Beirsdorf_randomfilepath = data["BEIERSDORF"] 
else:
    print("The beiersdorf key does not exist in the config file. Assigining none")
    Beirsdorf_randomfilepath = None
    # create the log file for logging missing data
    dummy_log_file = f'{Test_type}_missing.txt'

    # create the an empty "laylout" excel file
    excelfile = f'Layout_{studynumber}{Test_type}.xlsx'

# header handles
header = r"""\RequirePackage{pdf14}
\documentclass[a4paper]{scrartcl}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[utf8]{inputenc}
\usepackage{hyperxmp}
\usepackage[export]{adjustbox}
\usepackage{fancyhdr}
\usepackage[left=1.00cm,right=1.00cm,top=0.50cm,bottom=1.50cm,headheight=1.50cm,headsep=0.5cm,footskip=0.5cm,includeheadfoot]{geometry}
\setlength{\parindent}{0cm}
\usepackage[pdfa]{hyperref}
\def\arraystretch{0.9}
\usepackage{grffile}
\usepackage{pdf14}

\immediate\pdfobj stream attr{/N 3}  file{%s}
\pdfcatalog{/OutputIntents [ <<
/Type /OutputIntent
/S/GTS_PDFA1
/DestOutputProfile \the\pdflastobj\space 0 R
/OutputConditionIdentifier (%s)
/Info(%s)
>> ]
}
""" % (colorfile, colorname, colorname)

header_draft = r"""\RequirePackage{pdf14}
\documentclass[a4paper, draft]{scrartcl}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[utf8]{inputenc}
\usepackage{hyperxmp}
\usepackage[export]{adjustbox}
\usepackage{fancyhdr}
\usepackage[left=1.00cm,right=1.00cm,top=0.50cm,bottom=1.50cm,headheight=1.50cm,headsep=0.5cm,footskip=0.5cm,includeheadfoot]{geometry}
\setlength{\parindent}{0cm}
\usepackage[pdfa]{hyperref}
\def\arraystretch{0.9}
\usepackage{grffile}

\immediate\pdfobj stream attr{/N 3}  file{%s}
\pdfcatalog{/OutputIntents [ <<
/Type /OutputIntent
/S/GTS_PDFA1
/DestOutputProfile \the\pdflastobj\space 0 R
/OutputConditionIdentifier (%s)
/Info(%s)
>> ]
}
""" % (colorfile, colorname, colorname)

# No marketing handle
if No_marketing == "True":
    pagestyle_no_marketing =r"""\pagestyle{fancy}
    \lhead{%s}
    \rhead{\includegraphics[scale=0.5]{C:/Coding_Projekts/JB_py/Dev_Project5_ExcelDataToPDFconverter_Complete/source/bin/SGSProderm.jpg}}
    \lfoot{Confidential \\ Do not use for marketing purposes \\ \tiny For legal reasons, the images provided must not be used for marketing purposes.}
    \cfoot{%s\\ Image Overview}
    \rfoot{Page \thepage}
    """ % (r"SGS proderm GmbH", studynumber.replace('_', '\\_') )
else:
    pagestyle = r"""\pagestyle{fancy}
    \lhead{%s}
    \rhead{\includegraphics[scale=0.5]{C:/Coding_Projekts/JB_py/Dev_Project5_ExcelDataToPDFconverter_Complete/source/bin/SGSProderm.jpg}}
    \lfoot{Confidential \\ Vertraulich}
    \cfoot{%s\\ Image Overview}
    \rfoot{Page \thepage}
    """ % (r"SGS proderm GmbH", studynumber.replace('_', '\\_') )

hypersetup = r"""\hypersetup{pdftitle={Image Export},
pdfauthor={User},
pdfauthortitle={CIO},
pdfcopyright={Copyright (C) 2015, proderm},
pdfsubject={Image Overview},
pdfkeywords={image, overview},
pdflicenseurl={none},
pdfcaptionwriter={KPW},
pdfcontactaddress={Kiebitzweg 2},
pdfcontactcity={Schenefeld},
pdfcontactpostcode={22869},
pdfcontactcountry={Germany},
pdfcontactemail={info@proderm.de},
pdfcontacturl={http://www.proderm.de},
pdflang={en},
bookmarksopen=true,
bookmarksopenlevel=3,
hypertexnames=false,
linktocpage=true,
plainpages=false,
breaklinks}
"""
