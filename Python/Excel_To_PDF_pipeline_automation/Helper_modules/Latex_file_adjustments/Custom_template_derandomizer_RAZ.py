import re
import os

PATH_TO_ORIG_FILE = r"D:\STUDIES\22.0411_22.0428_RAZ\22.0411-33\pdf_final\images_tif\jpg_converted\jpg_small50\Test_results\22.0411-33_TestCustom_01_jpg__2022-12-21_12-49-35\22.0411-33_TestCustom_01_image_overview_final.tex"
PATH_TO_RDM_FILE = r"D:\Code\Software_test_sample_data\Dev_Proj_5__ExcelToPdfConverter\RDM_files\Random_22.0411-33_RAZ.txt"


rootPath = "\\".join(PATH_TO_ORIG_FILE.split("\\")[:-1])
newLatexFileName = PATH_TO_ORIG_FILE.split("\\")[-1].replace(".tex","_deRandomizedtest.tex")
PATH_TO_NEW_FILE = os.path.join(rootPath, newLatexFileName)

MASK_subj = r"Subject ([0-9]*)"


Mask_site = r"F[0-9]{2}"
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
        temp_dict = {}
        temp_element = line.split("\t")
        temp_dict["F01"] = temp_element[1]
        temp_dict["F02"] = temp_element[2]
        temp_dict["F03"] = temp_element[3]
        temp_dict["F04"] = temp_element[4].replace("\n", "")
        sorted_temp_dict = sorted(temp_dict.items(), key=lambda x: x[1])   
        rand_dict[temp_element[0]] = sorted_temp_dict
        
    if len(rand_dict) != 0:
        return rand_dict
    else:
        raise EmptyObjectReturnedError("Empty dict = please check filenames or fileext!")


def derandomize_Visia_Latexfile(rand_dict, path_to_latex_file, path_to_new_latex_file, subj_mask, site_mask):

    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    with open(path_to_new_latex_file, "w") as NewLatexF:
        change_flag = 0
        
        for n, line in enumerate(LFlines):
            match = search_text(subj_mask, line)
            
            if match != None:
                rand_tuple_list = rand_dict[match.group(1).lstrip("0")]
                if ("Subject" in match.group(0)) and (rand_tuple_list[0][0] !="F01" or rand_tuple_list[1][0] !="F02" or rand_tuple_list[2][0] !="F03" or rand_tuple_list[3][0] !="F04"):
                    change_flag = 1
                    NewLatexF.write(line) # write teh subject line to new latex file
                    continue

            if change_flag == 1:

                if "raisebox" in line or "tiny" in line:
                    if "F01" in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace("F01", rand_tuple_list[0][0])
                        NewLatexF.write(mod_line)
                        continue
                    if "F02" in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace("F02", rand_tuple_list[1][0])
                        NewLatexF.write(mod_line)
                        continue
                    if "F03" in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace("F03", rand_tuple_list[2][0])
                        NewLatexF.write(mod_line)
                        continue
                    if "F04" in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace("F04", rand_tuple_list[3][0])
                        NewLatexF.write(mod_line)
                        continue                                                            

            if "newpage" in line:
                change_flag = 0
                NewLatexF.write(line)
                continue
            else:
                NewLatexF.write(line)
    return NewLatexF

def main():

    RandDict = extract_randomization_scheme_from_rndmfile(PATH_TO_RDM_FILE)
    derandomize_Visia_Latexfile(RandDict, PATH_TO_ORIG_FILE, PATH_TO_NEW_FILE, MASK_subj, Mask_site)

if __name__ == "__main__":
    main()
    input("Press Enter to continue...")