import re
import os

PATH_TO_ORIG_FILE = r"D:\STUDIES\22.0340-51_scratch_woundheal\PDF\images_val\jpg_small50\Test_results\22.0340-51_TestCustom_03_jpg__2022-11-07_15-14-11\22.0340-51_TestCustom_03_image_overview_final.tex"
PATH_TO_RDM_FILE = r"D:\Code\Software_test_sample_data\Dev_Proj_5__ExcelToPdfConverter\RDM_files\Random_22.0340-51.txt"
# PATH_TO_PDF_IMGS = r"D:\STUDIES\22.0308_Shaving\imgs\Jpgs_clean\jpg_small50"
# FILE_EXT = ".jpg"
rootPath = "\\".join(PATH_TO_ORIG_FILE.split("\\")[:-1])
newLatexFileName = PATH_TO_ORIG_FILE.split("\\")[-1].replace(".tex","_deRandomized.tex")
PATH_TO_NEW_FILE = os.path.join(rootPath, newLatexFileName)

MASK_subj = r"Subject ([0-9]*)"
std_code_dict = {"A" : "F01",
                "B" : "F02",
                "C": "F03",
                "D" : "F04"}
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
        temp_element = line.split("\t")
        rand_dict[temp_element[0]] = (temp_element[1], temp_element[2],temp_element[3],temp_element[4].replace("\n", ""))
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

def derandomize_Visia_Latexfile(rand_dict, path_to_latex_file, path_to_new_latex_file, subj_mask, site_mask):

    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    with open(path_to_new_latex_file, "w") as NewLatexF:
        change_flag = 0
        
        for n, line in enumerate(LFlines):
            match = search_text(subj_mask, line)
            
            if match != None:
                rand_tuple = rand_dict[match.group(1).lstrip("0")]
                if ("Subject" in match.group(0)) and (rand_tuple != ("A", "B", "C", "D")):
                    change_flag = 1
                    NewLatexF.write(line) # write teh subject line to new latex file
                    continue
            # if change_flag == 1:
            #     count = 0
            #     if "Site" in line: 

            #         mod_line = line.replace("Site", "Site "+ rand_tuple[count])
            #         count += 1
            #         if count == len(rand_tuple)-1:

            #         NewLatexF.write(mod_line)
            #         continue   
            if change_flag == 1:
                if "Site" in line:
                    site_id = [k for k, v in std_code_dict.items() if v == re.search(site_mask, LFlines[n+1]).group(0)][0]
                    index = rand_tuple.index(site_id)
                    mod_line = line.replace("Site", "Product "+ rand_tuple[index])
                    NewLatexF.write(mod_line)
                    continue

                if "raisebox" in line or "tiny" in line:
                    if std_code_dict["A"] in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace(std_code_dict["A"], std_code_dict[rand_tuple[0]])
                        NewLatexF.write(mod_line)
                        continue
                    if std_code_dict["B"] in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace(std_code_dict["B"], std_code_dict[rand_tuple[1]])
                        NewLatexF.write(mod_line)
                        continue
                    if std_code_dict["C"] in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace(std_code_dict["C"], std_code_dict[rand_tuple[2]])
                        NewLatexF.write(mod_line)
                        continue
                    if std_code_dict["D"] in line:
                        # split_line = line.split("&&")
                        mod_line = line.replace(std_code_dict["D"], std_code_dict[rand_tuple[3]])
                        NewLatexF.write(mod_line)
                        continue                                                            

            if "newpage" in line:
                change_flag = 0
                NewLatexF.write(line)
                continue
            else:
                NewLatexF.write(line)
    return NewLatexF

def sort_according_to_product(path_to_new_latex_file, site_mask, subj_mask, rand_dict):
    
    with open(path_to_new_latex_file, "r+") as latexF:
        LFlines = latexF.readlines()
    
    for n, line in enumerate(LFlines):
        match = search_text(subj_mask, line)
        if match != None and ("Subject" in match.group(0)):
            rand_tuple = rand_dict[match.group(1).lstrip("0")]

    with open(path_to_new_latex_file, "w") as NewLatexF:
        for n, line in enumerate(LFlines):
            match = search_text(subj_mask, line)
            if match != None and ("Subject" in match.group(0)):
                rand_tuple = rand_dict[match.group(1).lstrip("0")]
                NewLatexF.write(line)
                continue

            if "Site" in line:
                site_id = [k for k, v in std_code_dict.items() if v == re.search(site_mask, LFlines[n+1]).group(0)][0]
                index = rand_tuple.index(site_id)
                mod_line = line.replace("Site", "Site "+ rand_tuple[index])
                NewLatexF.write(mod_line)
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