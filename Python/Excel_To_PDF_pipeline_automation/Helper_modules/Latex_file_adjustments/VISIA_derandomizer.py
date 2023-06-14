import re
import os

PATH_TO_ORIG_FILE = r"D:\STUDIES\22.0308_Shaving\imgs\Jpgs_clean\jpg_small50\Test_results\22.0308-23_TestCustom_09_jpg__2022-11-04_11-30-21\22.0308-23_TestCustom_09_image_overview_finalmodsubjfieldsKopie.tex"
PATH_TO_RDM_FILE = r"D:\Code\Software_test_sample_data\Dev_Proj_5__ExcelToPdfConverter\RDM_files\Random_22.0308-23.txt"
# PATH_TO_PDF_IMGS = r"D:\STUDIES\22.0308_Shaving\imgs\Jpgs_clean\jpg_small50"
# FILE_EXT = ".jpg"
rootPath = "\\".join(PATH_TO_ORIG_FILE.split("\\")[:-1])
newLatexFileName = PATH_TO_ORIG_FILE.split("\\")[-1].replace(".tex","_deRandomized.tex")
PATH_TO_NEW_FILE = os.path.join(rootPath, newLatexFileName)

MASK = r"Subject ([0-9]*)"

class EmptyObjectReturnedError(Exception):
    pass

def search_text(mask_pattern, line_to_search):
    try:
        return re.search(mask_pattern, line_to_search)
    except AttributeError:
        return None

def extract_randomization_scheme_from_rndmfile(random_file_path):
    with open(random_file_path, "r") as randF:
        rand_lines =  randF.readlines()[5:]
    rand_dict = {}
    
    for line in rand_lines:
        temp_element = line.split("\t")
        rand_dict[temp_element[0]] = (temp_element[1], temp_element[2].replace("\n", ""))
    if len(rand_dict) != 0:
        return rand_dict
    else:
        raise EmptyObjectReturnedError("Empty dict = please check filenames or fileext!")

# def create_image_paths_dict(path_to_PDF_images, file_ext):
#     imgs_path_dict = {}
#     for fn in os.listdir(path_to_PDF_images):
#         if fn.endswith(file_ext):
#             imgs_path_dict[fn] = os.path.join(path_to_PDF_images, fn)
#     if len(imgs_path_dict) != 0:
#         return imgs_path_dict
#     else:
#         raise EmptyObjectReturnedError("Empty dict = please check filenames or fileext!")



def derandomize_Visia_Latexfile(rand_dict, path_to_latex_file, path_to_new_latex_file, mask):

    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    with open(path_to_new_latex_file, "w") as NewLatexF:
        flag = 0
        for line in LFlines:
            match = search_text(mask, line)
            if match != None:
                if ("Subject" in match.group(0)) and (rand_dict[match.group(1).lstrip("0")] == ("B", "A")):
                    flag = 1
                    NewLatexF.write(line)
                    continue
            if flag == 1:
                if "raisebox" in line or "tiny" in line:
                    split_line = line.split("&&")
                    first_replacement = split_line[0].replace("Right View", "Left View")
                    second_replacement = split_line[1].replace("Left View", "Right View")
                    mod_line = "&&".join([first_replacement, second_replacement])
                    NewLatexF.write(mod_line)
                    continue

            if "newpage" in line:
                flag = 0
                NewLatexF.write(line)
                continue
            else:
                NewLatexF.write(line)

def main():

    RandDict = extract_randomization_scheme_from_rndmfile(PATH_TO_RDM_FILE)
    derandomize_Visia_Latexfile(RandDict, PATH_TO_ORIG_FILE, PATH_TO_NEW_FILE, MASK)

if __name__ == "__main__":
    main()
    input("Press Enter to continue...")