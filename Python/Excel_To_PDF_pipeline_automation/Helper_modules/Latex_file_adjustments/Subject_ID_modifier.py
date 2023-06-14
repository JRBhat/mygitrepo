import re
import os

PATH_TO_ORIG_FILE = r"D:\STUDIES\22.0308_Shaving\imgs\Jpgs_clean\jpg_small50\par-pol\jpg_small50\Test_results\22.0308-23_TestTranspose_with_Randomization_06_jpg__2022-11-08_15-21-08\22.0308-23_TestTranspose_with_Randomization_06_image_overview_finalKopie.tex"
MASK = r"Subject [0-9]*"
rootPath = "\\".join(PATH_TO_ORIG_FILE.split("\\")[:-1])
newLatexFileName = PATH_TO_ORIG_FILE.split("\\")[-1].replace(".tex","_modifiedSubjFields.tex")
PATH_TO_NEW_FILE = os.path.join(rootPath, newLatexFileName)
# LIST_OF_APPENDICIES = ["Day 1 Parallel Polarized", "Day 8 Parallel Polarized","Day 1 Standard 1","Day 8 Standard 1", 
                                                # "Day 1 Standard 2","Day 8 Standard 2", "Day 1 Cross-Polarized", "Day 8 Cross-Polarized"]
                                                
LIST_OF_APPENDICIES = ["Day 1 Parallel Polarized", "Day 8 Parallel Polarized"]
#regionhidden varname:TEXTTOAPPEND usecase:For simple case only                                    
#TEXT_TO_APPEND= r"APPENDEDSTRINGHERE"
#endregion

class MissingSubjectIDError(Exception):
    pass

def search_text(mask_pattern, line_to_search):
    try:
        return re.search(mask_pattern, line_to_search).group(0)
    except AttributeError:
        return None

def change_same_for_each_subject_id(path_to_latex_file, path_to_new_latex_file, text_to_append, mask):
    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    with open(path_to_new_latex_file, "w") as NewLatexF:
        for line in LFlines:
            match = search_text(mask, line)
            if match != None:
                replacement =  match + " " + text_to_append
                modf_line = line.replace(match, replacement)
                NewLatexF.write(modf_line)
            else:
                NewLatexF.write(line)

def change_custom_for_each_subject_id(path_to_latex_file, path_to_new_latex_file, mask, list_of_appendices):
    
    count = 0
    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    check = 0
    for line in LFlines:
        match = search_text(mask, line)
        if match != None:
            check = check + 1

    if check % len(list_of_appendices) == 0:
        with open(path_to_new_latex_file, "w") as NewLatexF:
            for line in LFlines:
                match = search_text(mask, line)
                if match != None:
                    if count > len(list_of_appendices)-1:
                        count = 0
                    replacement =  match + " " + list_of_appendices[count]
                    count = count + 1
                    modf_line = line.replace(match, replacement)
                    NewLatexF.write(modf_line)
                else:
                    NewLatexF.write(line)
    else:
        raise MissingSubjectIDError(f"{check} !=% {len(list_of_appendices)} Check the number of subject Ids in the Latex file again. \n \
                                They don't seem to be divisible by the length of the list of appendices.")

def main():
    # change_same_for_each_subject_id(PATH_TO_ORIG_FILE, PATH_TO_NEW_FILE, TEXT_TO_APPEND, MASK)
    change_custom_for_each_subject_id(PATH_TO_ORIG_FILE, PATH_TO_NEW_FILE, MASK, LIST_OF_APPENDICIES)

if __name__ == "__main__":
    main()
    input("Press Enter to continue...")