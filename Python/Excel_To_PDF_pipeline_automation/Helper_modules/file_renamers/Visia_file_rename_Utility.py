from operator import ne
import os
import re
import json
import sys
import time

class filenameError(Exception):
    pass

def reverse_renaming(json_log_file_path, raw_path, file_ext):
    with open(json_log_file_path, "r") as jsonF:
        orig_dict = json.load(jsonF)
    
    rawList = os.listdir(raw_path)
    for k in orig_dict.keys():
        try:
            mod_fn = list(filter(lambda x: k == x and file_ext in x, rawList))
            orig_name = orig_dict[mod_fn[0]]
            print(orig_name)
            os.rename(os.path.join(raw_path, mod_fn[0]), os.path.join(raw_path, orig_name))
        except (AttributeError,IndexError) as err:
            print(err)
            sys.exit(1)
    print("File reconstruction of all files to original naming scheme...without errors ")
    del_flag = input("Delete Json log file?(y/n)")
    if del_flag == "y":
        os.remove(json_log_file_path)

def rename_visia_files(actual_path, studyID):
    
    file_ext = ".jpg"
    rgx_mask = r"([0-9]*)_(T[0-9]{2})_([a-zA-Z \-]*)_([a-zA-Z1-2 \-]*)"
    
    side_list= []
    rgx = re.compile(rgx_mask)

    
    log_path = os.path.join(actual_path, f"{studyID}_LOG_VisiaRenamedFiles.json")

    for fn in os.listdir(actual_path):
        if fn.endswith(file_ext):    
            try:
                side_list.append(re.search(rgx, fn).group(3))
            except:
                raise filenameError(f"Check side id again for {fn} with mask {rgx_mask}")

    side_list = set(side_list)
    # associates a side number to the side names # visia only has 3 views
    sides_key_dict = {"Right View": 1, 
                      "Front View": 2, 
                      "Left View": 3}
      
    sides_counter_list = ["F01", "F02", "F03"]

    sorted_key = [(x , sides_key_dict[x]) for x in side_list]
    sorted_side_keys = sorted(sorted_key, key= lambda x: x[1])

    side_dict = {}
    for n, ele in enumerate(sorted_side_keys):
        side_dict[ele[0]] = sides_counter_list[n]

    # assoicates an abbreviated string to the lighting modes
    light_dict = {"Standard 1":"STD1", 
                "Standard 2":"STD2", 
                "UV-NF":"UVNF",
                "Cross-Polarized": "XPOL", 
                "Parallel-Polarized":"PPOL"}

    log_dict = {}

    for fn in os.listdir(actual_path):
        
        if fn.endswith(file_ext):
            try:
                orig_subj_num = re.search(rgx, fn).group(1)
            except:
                raise filenameError(f"check Subj of {fn} for mask {rgx_mask}")
            if len(orig_subj_num) == 3:
                subjHead = "S" + orig_subj_num
            elif len(orig_subj_num) == 2:
                subjHead = "S0" + orig_subj_num
            elif len(orig_subj_num) == 1:
                subjHead = "S00" + orig_subj_num
            try:
                timeHead = re.search(rgx, fn).group(2)
            except:
                raise filenameError(f"check timep of {fn} for mask {rgx_mask}")                
            try:
                sideHead = side_dict[re.search(rgx, fn).group(3)]
            except:
                raise filenameError(f"check sideid of {fn} for mask {rgx_mask}")                
            try:
                lightHead = light_dict[re.search(rgx, fn).group(4)]
            except:
                raise filenameError(f"check light id of {fn} for mask {rgx_mask}")                

            # new filename
            new_fn = subjHead + sideHead +timeHead + lightHead + "--"+fn #+"_"+idHead 
            print(new_fn)
            # log all changes to a tuple and save it in a dict
            log_dict[new_fn] = fn
                   
    #rename - code kept outside first loop ; renaming occurs only if the dict is properly written without errors
    print("No errors found during the dictionary creation process.... Now renaming")
    print("Please Wait until program ENDS...")
    time.sleep(3)
    for new_fn, fn in log_dict.items():
        os.rename(os.path.join(actual_path, fn), os.path.join(actual_path, new_fn))
    
    # dump dict to dict for logging
    with open(log_path, "w") as logJson:
        json.dump(log_dict, logJson)

    print("Renaming finished; changes stored in json log file")
    
# uncomment for testing or reverse renaming
# if __name__ == "__main__":
    # studynum = "23.0019_Antiakne"
    # PATH = r"D:\Code\Software_test_sample_data\Dev_Proj_5__ExcelToPdfConverter\Visia_renamer_test_samples\23.0019-54_antiakne"
    # rename_visia_files(PATH, studynum)