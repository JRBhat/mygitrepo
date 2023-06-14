'''
This module contains all the independent functions used regularly in various templates and files. This
Mainly intended for imporving clarity, consistency and simplifying debugging, while also reducing code redundnacy and possible bug induction due to spaghetti code.
'''
import subprocess
import os
import re
from string import digits
from ImageAnalysis import ColorConversion as cc
from ImageAnalysis import Util
import cv2 


def check_for_next_page_request(rownames, *args):
    """
    check_for_next_page_request 
    """
    for each_ele in rownames:
        if  'next' in each_ele:
            get_indx_whr_next_found = rownames.index(each_ele)
            args[0].insert(get_indx_whr_next_found, 'next')
    return args[0] 

def reverse_n_replace(item, replace_from, replace_to):
    """
    reverse_n_replace 
    """    
    run_1 = item[::-1]
    run_2 = run_1.replace(replace_from, replace_to, 1)
    print(run_2[::-1])
    return run_2[::-1]

def get_specefic_subj_value(fs, fa, ft, listx):
    """
    get_specefic_subj_value 
    """    
    try:
        l = list(filter(lambda x : fs in x and fa in x and ft in x, listx))
        return l[0]
    except IndexError:
        print(f"{fs}{fa}{ft} image not available")
        return 'N.A.'

def create_column_block(outer_iterator, inner_iterator):
    """
    create_column_block in latex document
    """    
    block = r"&"
    for ele_outer in outer_iterator:
        block += (ele_outer + r"&&")
    for ele_inner in inner_iterator:
        if ele_inner in block:
            block = block.replace(ele_inner, "")
    return reverse_n_replace(block, r"&&", r"\\").translate({ord('$'): None}).replace("*", "")    

def insert_new_page(colnames, col_name_block, subj):
    """
    insert_new_page 
    """
    return r"\end{tabular}" + "\n" + r"\newpage" +"\n" + r"{\Large %s}\\[0.2cm]"%(f"Subject {subj[-2:]}") + "\n" + \
        r"\begin{tabular}{%s}"%("".join(['cc']*(len(colnames)))) + "\n" + col_name_block

def insert_lax_rowname_path_line(subj, max_height, max_width, path_extract_func, *args):
    """
    insert_lax_rowname_path_line 
    """
    try:    
        return r"\raisebox{-.5\height}{\includegraphics[max height=" + rf"{max_height}" + \
            r"\textheight,max width="+ rf"{max_width}" + r"\textwidth]{" + \
                path_extract_func(subj, *args).replace('\\',"/") + r"}}" + r"&&" + ""
    except AttributeError:
        print(""" - function Returns None. Possible causes might be:
                - you forgot to copy the first subj template to the modified template worksheet, 
                - images N.A. or wrongly numbered, 
                - regex does not match complete filename, incomplete path will be printed to the dictionary""")
        exit(1)

def insert_random_lax_rowname_path_line(spit_out_val, max_height, max_width):
    """
    insert_random_lax_rowname_path_line 
    """    
    return r"\raisebox{-.5\height}{\includegraphics[max height=" + rf"{max_height}" + \
            r"\textheight,max width="+ rf"{max_width}" + r"\textwidth]{" + \
              spit_out_val[0].replace('\\',"/") + r"}}" + r"&&" + ""

def insert_random_img_filename_line(spit_out_val):
    """
    insert_img_filename_line 
    """        
    raw_filename = spit_out_val[0].split("\\")[-1][:-4]
    if "dummy" in raw_filename:
        raw_filename = "missing image"

    return r"{\tiny " + raw_filename.replace('_', '\\_') + r"}" + r"&&"

def insert_img_filename_line(subj, path_extract_func, *args): #TODO: If dummy found in name; then replace with missing image
    """
    insert_img_filename_line 
    """
    raw_filename = path_extract_func(subj, *args).split("\\")[-1][:-4]
    
    if "dummy" in raw_filename:
        raw_filename = "missing image"
    return r"{\tiny " + raw_filename.replace('_', '\\_') + r"}" + r"&&"      
    

# Used only by Custom_template, (add further usages if any),...
def get_matrix(ws):
    """
    get_matrix 
    """    
    # # Get new image matrix
    img_list_pg1 = []
    for ro_indx, _ in enumerate(ws.rows, start=3):
        img_list_pg1.append([])
        for col_indx, _ in enumerate(ws.columns, start=2):
            if ws.cell(row=ro_indx, column=col_indx).value == None:
                break
            x = ws.cell(row=ro_indx, column=col_indx).value
            img_list_pg1[-1].append(x)
    img_list_pg1 = list(filter(lambda x : x != [], [x for x in img_list_pg1]))# cleaning out empty '[]' from list
    return img_list_pg1
    # print(img_list_pg1) # TODO: possible adjustment for randomization ; values after modifing layout in custom tempalte

# Used only by Custom_template, (add further usages if any),...
def get_pathvalue_from_main_dict(subjno, mainMappingDict, elmt):
    """
    get_pathvalue_from_main_dict [summary]

        :param subjno: [description]
        :type subjno: [type]
        :param mainMappingDict: [description]
        :type mainMappingDict: [type]
        :param elmt: [description]
        :type elmt: [type]
        :return: [description]
        :rtype: [type]
    """    
    for key in mainMappingDict.keys():
        sub_elmt = elmt.replace("S$Sub_id$", subjno)
        if sub_elmt == key[0] and subjno == key[1]:
            return mainMappingDict[key]

def validate_random_file(random_file_path, subject_list):
    # create a set for the subject list A
    subj_list = []
    for sID in subject_list:
        if sID[-2:][0] != "0": # if the first digit of the last two numbers of subj id are non-zero 
            subj_list.append(sID[-2:])
        elif sID[-2:][0] == "0": # if the first digit of the last two numbers of subj id is zero
            subj_list.append(sID[-1])
        else:
            raise ValueError

    #create a set for the randomfile B
    rand_list = []
    with open(random_file_path) as f:
        lines = f.readlines()
    interesting_lines = lines[5:]
    for line in interesting_lines:
        rand_list.append(line.split("\t")[0])
    
    # calculate difference, i.e. B - A that gives the subject ids that need to be removed from the random file
    if len(rand_list) > len(subj_list):
        diff_set = set(rand_list) - set(subj_list)
        print(f"differences found: {diff_set}")
        for diff in diff_set:
            for line in interesting_lines:
                if diff == line.split("\t")[0]:
                    interesting_lines.remove(line)
    else:
        print("No discrepancies found in random file.")
    
    # print(clean_list[5:])
    clean_list = []
    for l in interesting_lines:
        stripped = re.sub(r"\W", "", l)
        remove_digits = str.maketrans('', '', digits)
        str_only = stripped.translate(remove_digits)
        clean_list.append(str_only)
    return clean_list

def create_Latex_document(colnames, rownames, sub_loop_sorted_list, max_H, max_W, *args, Type="Standard", RandomList=None): # RandomList=MyRandomList, outer_loop_parameter_list, inner_loop_parameter_list, filepaths, filenamelist):
    """
    create_Latex_document 
    """

    lax_document = [] 
    lax_page = {}

    lax_page['heading'] = []
    lax_page['table'] = []  

    if Type == "Standard":

        checked_list = check_for_next_page_request(rownames, args[0]) # args[0] was previously outer_loop_parameter_list aka area_sorted

        col_name_block = create_column_block(colnames, args[1])  # args[1] was previously inner_loop_parameter_list aka time_sorted


        for subj in sub_loop_sorted_list:

            lax_page['heading'].append(f"Subject {subj[-2:]}")                                                  # similar to {\Large Subject 1}\\[0.2cm]
            lax_page['col_name_block'] = col_name_block                                                         # similar to column name line --> &Visit 3 (Part II)&&Visit 3 (Part III)\\

        # Lists all paths and filenames of currently selected subject no 
            current_subj_paths = list(filter(lambda x : subj in x, args[2]))         #args[2] was previously filepaths
            current_subj_names = list(filter(lambda x : subj in x, args[3]))       #args[3] was previously filenamelist

            # Initialize variables for the latex tables
            lax_sub_table = []
            lax_rowname_path_line = r"\raisebox{-.5\height}{\rotatebox{90}{"
            lax_img_filename_line = r"&"
            
            for a in checked_list:   
                new_page_alert= 0 # flag indicates a new page insertion
                if a == 'next':
                    new_page = insert_new_page(colnames, col_name_block, subj)
                    lax_sub_table.append(new_page)
                    new_page_alert = 1    # similar to column name line --> &Visit 3 (Part II)&&Visit 3 (Part III)\\
                    continue                                                         
                
                lax_rowname_path_line += list(filter(lambda x : a in x, rownames))[0].translate({ord('$') : None}).replace(a, "").replace('*', "") + r"}}&"
                
                for t in args[1]: # args[1] was previously inner_loop_parameter_list
                    if RandomList != None:
                        spit_out_from_iterator = next(RandomList)
                        lax_rowname_path_line += insert_random_lax_rowname_path_line(spit_out_from_iterator, max_H, max_W)
                        lax_img_filename_line += insert_random_img_filename_line(spit_out_from_iterator)
                                                        # modify latex code when end of list is reached

                        if t == args[1][-1]: # args[1] was previously inner_loop_parameter_list
                            lax_sub_table.append(reverse_n_replace(lax_rowname_path_line, r"&&", r"\\" ))
                            lax_sub_table.append(reverse_n_replace(lax_img_filename_line, r"&&", r"\\" ))
                            lax_rowname_path_line = r"\raisebox{-.5\height}{\rotatebox{90}{"
                            lax_img_filename_line = r"&"


                    else:    
                        lax_rowname_path_line += insert_lax_rowname_path_line(subj, max_H, max_W, get_specefic_subj_value, a, t, current_subj_paths)

                        lax_img_filename_line += insert_img_filename_line(subj, get_specefic_subj_value, a, t, current_subj_names)

                        # modify latex code when end of list is reached
                        if t == args[1][-1]: # args[1] was previously inner_loop_parameter_list
                            lax_sub_table.append(reverse_n_replace(lax_rowname_path_line, r"&&", r"\\" ))
                            lax_sub_table.append(reverse_n_replace(lax_img_filename_line, r"&&", r"\\" ))

                            lax_rowname_path_line = r"\raisebox{-.5\height}{\rotatebox{90}{"
                            lax_img_filename_line = r"&"

            lax_page['table'].append(lax_sub_table)
            lax_document.append(lax_page)
        lax_document[-1]['end'] = True

        return lax_document, new_page_alert

    elif Type == "Custom":

         # args[0] was previously main_mapping_dict

        lax_document = [] 
        lax_page = {}

        img_matrix = get_matrix(args[2]) # args[2] was previously ws # 

        img_matrix = check_for_next_page_request(rownames, img_matrix)

        col_name_block = create_column_block(colnames, args[1])

        lax_page['heading'] = []
        lax_page['table'] = []    

        for subj in sub_loop_sorted_list:

            lax_page['heading'].append(f"Subject {subj[-2:]}")                                                     
            lax_page['col_name_block'] = col_name_block                                                        

            lax_sub_table = []
            lax_rowname_path_line = r"\raisebox{-.5\height}{\rotatebox{90}{"
            lax_img_filename_line = r"&"

            for r_indx, r_val in enumerate(img_matrix):
                new_page_alert= 0                                                     
                if r_val == 'next':
                    new_page = insert_new_page(colnames, col_name_block, subj)
                    lax_sub_table.append(new_page)
                    new_page_alert = 1    # similar to column name line --> &Visit 3 (Part II)&&Visit 3 (Part III)\\
                    continue

                lax_rowname_path_line += rownames[r_indx] + r"}}&"
                for element in r_val:
                    if RandomList != None:
                        spit_out_from_iterator = next(RandomList)
                        lax_rowname_path_line += insert_random_lax_rowname_path_line(spit_out_from_iterator, max_H, max_W)
                        lax_img_filename_line += insert_random_img_filename_line(spit_out_from_iterator)

                    else:
                        lax_rowname_path_line += insert_lax_rowname_path_line(subj, max_H, max_W, get_pathvalue_from_main_dict, args[3], element) # args[3] was previously Main_Mapping_dict; element is object read from image matrix
                        lax_img_filename_line += insert_img_filename_line(subj, get_pathvalue_from_main_dict, args[3], element) # args[3] was previously Main_Mapping_dict; element is object read from image matrix

                    # convert the above lists into a function that returns 'missin or N/A when an image is not found 
                        # then call the funciton within the lines above

                lax_sub_table.append(reverse_n_replace(lax_rowname_path_line, r"&&", r"\\" ))
                lax_sub_table.append(reverse_n_replace(lax_img_filename_line, r"&&", r"\\" ))
                lax_rowname_path_line = r"\raisebox{-.5\height}{\rotatebox{90}{"
                lax_img_filename_line = r"&"

            lax_page['table'].append(lax_sub_table)
            lax_document.append(lax_page)
        lax_document[-1]['end'] = True

    return lax_document, new_page_alert       

def get_colorprofile_specs(path_to_images):
    """
    check_image_type_and_colorprofile 
    """
    files_list = os.listdir(path_to_images)
    # check for jpgs
    jpg_found_list = []
    for f in files_list:
        if f.endswith("jpg") and isinstance(f, str):
            jpg_found_list.append(f)
            type = "jpg"
            filepath = os.path.join(path_to_images, f)
            colorspace_tuple = cc.get_colorprofile(filepath)
            cfile, cname = color_profile_elements(colorspace_tuple)
            return cfile, cname, type
            
    # # if no jpgs found, check for pngs or tifs
    png_found_list = []
    if len(jpg_found_list) == 0:
        for f in files_list:
            if f.endswith("png") and isinstance(f, str):
                png_found_list.append(f)
                type = "png"
                filepath = os.path.join(path_to_images, f)
                colorspace_tuple = cc.get_colorprofile(filepath)
                cfile, cname = color_profile_elements(colorspace_tuple)
                return cfile, cname, type
                       
    if len(png_found_list) == 0 and len(jpg_found_list)== 0:
        for f in files_list:
            if (f.endswith("tiff") or f.endswith("tif") or f.endswith("TIF")) and isinstance(f, str):
                type = "tif"
                filepath = os.path.join(path_to_images, f)
                colorspace_tuple = cc.get_colorprofile(filepath)
                cfile, cname = color_profile_elements(colorspace_tuple)
                return cfile, cname, type 
                                  
def color_profile_elements(colorspace_tuple):
    """
    get_color_profile 
    """
    try:
        if colorspace_tuple[0] == 'srgb':
            colorfile = 'C:/Coding_Projekts/JB_py/Dev_Project5_ExcelDataToPDFconverter_Complete/source/bin/sRGB_Color_Space_Profile.icm'
            colorname = "sRGB Color Space Profile"
            return colorfile, colorname 
        elif colorspace_tuple[0] == 'adobe':
            colorfile = 'C:/Coding_Projekts/JB_py/Dev_Project5_ExcelDataToPDFconverter_Complete/source/bin/AdobeRGB1998.icc'
            colorname = "adobeRGB Color Space Profile"
            return colorfile, colorname 
    except IndexError:
        print("No colorprofile found, applying default colorspace sRGB")
        colorfile = 'C:/Coding_Projekts/JB_py/Dev_Project5_ExcelDataToPDFconverter_Complete/source/bin/sRGB_Color_Space_Profile.icm'
        colorname = "sRGB Color Space Profile"
        return colorfile, colorname 

def convert_other_extensions_to_jpg(path, type):
    """
    convert_other_extensions_to_jpg 
    """
    if type == "png":
        return jpg_conversion_engine(type, path)

    elif type == "tif":
        return jpg_conversion_engine(type, path)

def jpg_conversion_engine(ext_type, path_to_be_converted):
    """
    jpg_conversion_engine 
    """    
    new_jpg_dir = os.path.join(path_to_be_converted, "jpg_converted")
    try:
        os.mkdir(new_jpg_dir)
    except FileExistsError:
        print("File already exists. So returning existing file. Checking whether resize file exists.")
        # new_jpg_list = [f for f in os.listdir(new_jpg_dir)]
        return resize_jpg(new_jpg_dir)
    # magick mogrify command for conversion to jpg
    proc = subprocess.Popen(f"d: && cd {path_to_be_converted} && magick mogrify -format jpg -depth 8 -path jpg_converted *.{ext_type}", shell=True)
    print("Converting images to JPG....")
    proc.wait()
    print("Conversion complete!")

    # validation of new path
    jpg_list = [f for f in os.listdir(new_jpg_dir)]
    if len(jpg_list) == 0:
        print(jpg_list)
        os.rmdir(new_jpg_dir)
    else:
        try:
            return resize_jpg(new_jpg_dir)
        except TypeError:
            print("Resize limit reached. Total file size less than 400MB and is well-adjusted for PDF use")
            print("Returning non-scaled converted jpgs")
            return new_jpg_dir, jpg_list
        
def resize_jpg(path):
    """
    resize_jpg 
    """    
    total_size = 0
    sub_1_count = 0
    check_list = []
    sub1_fileCheckMask = re.compile(r"S001F[0-9]{2}T[0-9]{2}")

    # gets all jpg images related to study
    for f in os.listdir(path): 
        if f.endswith("jpg") and isinstance(f, str):
            check_list.append(f)
            total_size += os.path.getsize(os.path.join(path, f))

    # gets all images for Subject1        
    for f in os.listdir(path):         
        try: 
            if f.endswith("jpg") and isinstance(sub1_fileCheckMask.search(f).group(0), str):
                sub_1_count += 1
        except AttributeError:
            break
    print(total_size/1024**2)
    print(sub_1_count)
    thresh = get_resize_percentage(total_size, sub_1_count)
    if thresh == None:
        print("File size less than 400MB")
        return TypeError
    else:
        small_jpg_dir = os.path.join(path, f"jpg_small{thresh[:-1] }")
        try:
            os.mkdir(small_jpg_dir)
        except FileExistsError:
            print("File already exists. So returning existing file. No further conversion needed.")
            small_jpg_list = [f for f in os.listdir(small_jpg_dir)]
            return small_jpg_dir, small_jpg_list
        
        # magick mogrify command for resizing jpgs
        proc = subprocess.Popen(f"d: && cd {path} &&magick mogrify -resize {thresh} -path jpg_small{thresh[:-1]} *.jpg", shell=True)
        print(f"Resizing images to {thresh} of jpgs....")
        proc.wait()
        print("Resize complete!")
        small_jpg_list = [f for f in os.listdir(small_jpg_dir)]
        if len(small_jpg_list) == 0:
            print(f"The folder {small_jpg_list} is empty")
            os.rmdir(small_jpg_dir)
        return small_jpg_dir, small_jpg_list

def get_resize_percentage(total_imgs_size, sub1_counter):
    """
    get_resize_percentage 
    """    
    my_dict = {500 : {1:"50%", 0.5:"45%", 0:"40%"}, 450 : {1:"45%", 0.5:"40%", 0:"30%"}, 400 : {1:"40%", 0.5:"35%", 0:"30%"}} 
    for limit in my_dict.keys():
        if (total_imgs_size/1024**2) >= limit:
            for threshold in my_dict[limit].keys():
                if (sub1_counter/10) >= threshold:
                    return my_dict[limit][threshold]
        else:
            for threshold in my_dict[limit].keys():
                if (sub1_counter/10) >= threshold:
                    return my_dict[limit][threshold]

def get_row_and_columns(ws, rownames_start, colnames_cell_row):

        """
        Reads the rows and columns from the 'Modify_Template_here' excel sheet and stores those values in the columns'
        corresponding lists.
        """
        
        # get row names
        rownames = []
        for ro_indx, _ in enumerate(ws.rows, start= rownames_start):
            x = ws.cell(row=ro_indx, column=1).value
            rownames.append(x)    
            if x == None:
                break
        rownames.remove(None)
        print(rownames)

        # get column names
        colnames = []
        for col_indx, _ in enumerate(ws.columns, start= 2):
            y = ws.cell(row=colnames_cell_row, column=col_indx).value
            colnames.append(y)    
            if y == None:
                break
        colnames.remove(None)
        print(colnames)
        return rownames, colnames

def extract_elements_from_regex_mask(source_to_extract_from, reg_mask):
    # Id_mask = re.compile(r"Subject [0-9]*")
    # Subj_id = Id_mask.search(l).group(0)

    reg_obj = re.compile(reg_mask)
    extracted_ele = reg_obj.search(source_to_extract_from).group(0)
    return extracted_ele

def create_blank_dummy(height, width, dummy_filepath):#, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    # image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    # color = tuple(reversed(rgb_color))
    # Fill image with color
    # image[:] = color
    image = cv2.imread(r"D:\Code\Codebase_Bha\Dev_Project5_ExcelDataToPDFconverter_Complete\source\bin\missing_pic.jpg")
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(dummy_filepath, resized)
    
def get_image_properties(image_path):
    im = cv2.imread(image_path)
    height = im.shape[0]
    width = im.shape[1]
    return height, width

def make_dict_with_barcode_as_keys(list_w_filepaths):
    
    Subj_mask = re.compile(r'S[0-9]{3}')
    Area_mask = re.compile(r'F[0-9]{2}')
    Time_mask = re.compile(r'T[0-9]{2}')
    barcode_dict = {}

    for fpath in list_w_filepaths:
        barcode_dict[(Subj_mask.search(fpath.split("\\")[-1]).group(0), 
                        Area_mask.search(fpath.split("\\")[-1]).group(0), 
                            Time_mask.search(fpath.split("\\")[-1]).group(0))] = fpath
    return barcode_dict

def get_barcode_groups(filepathList):
    Subj_mask = re.compile(r'S[0-9]{3}')
    Area_mask = re.compile(r'F[0-9]{2}')
    Time_mask = re.compile(r'T[0-9]{2}')

    subj_grp, area_grp, time_grp = [], [], []

    for fpath in filepathList:
        fpath_str_list = fpath.split('\\')

        subj_grp.append(Subj_mask.search(fpath_str_list[-1]).group(0))
        area_grp.append(Area_mask.search(fpath_str_list[-1]).group(0))
        time_grp.append(Time_mask.search(fpath_str_list[-1]).group(0))

    # sorting unique ids
    subj_grp_sorted = sorted(set(subj_grp))
    area_grp_sorted = sorted(set(area_grp))
    time_grp_sorted = sorted(set(time_grp))

    return subj_grp_sorted, area_grp_sorted, time_grp_sorted

def find_missing_barcodes(filepath_list):
    barc_dict = make_dict_with_barcode_as_keys(filepath_list)
    missing_list = []
    # creating the dict
    s_grp, a_grp, t_grp = get_barcode_groups(filepath_list)
    for subj in s_grp:
        for area in a_grp:
            for time in t_grp:
                if (subj, area, time) not in barc_dict.keys():
                    missing_list.append((subj, area, time))

    return missing_list

def replace_missing_barcodes_with_dummy(list_with_filepaths, visia_flag=0): 

    missing_list = find_missing_barcodes(list_with_filepaths)
    if len(missing_list) == 0:
        print("NO MISSING VALUES FOUND...None is returned")
        input("Press Enter to proceed...")
        return None

    print("Missing values found...Please check log file and adjust tex file accordingly.")
    input("Press Enter to proceed...")
    Subj_mask = re.compile(r'S[0-9]{3}')
    Area_mask = re.compile(r'F[0-9]{2}')
    Time_mask = re.compile(r'T[0-9]{2}')
    Light_mask = re.compile(r'(T[0-9]{2})([A-Za-z0-9]{4})')
    dummy_path_list = [] 
    for ids_tup in missing_list:

        first_path_as_sample = list_with_filepaths[0]
        image_name = first_path_as_sample.split("\\")[-1]
        old_subj_id = Subj_mask.search(image_name).group(0)
        old_area_id = Area_mask.search(image_name).group(0)
        old_time_id = Time_mask.search(image_name).group(0)

        if visia_flag == 1:
            light_codes = ["STD1", "STD2", "UVNF","XPOL", "PPOL"]
            for light_name in light_codes:
                old_light_id = Light_mask.search(image_name).group(2)
                dummy_name = image_name.replace(old_subj_id, ids_tup[0]).replace(old_area_id, ids_tup[1]).replace(old_time_id, ids_tup[2]).replace(old_light_id, light_name)
                dummy_path = first_path_as_sample.replace(image_name, dummy_name.replace(".jpg", "_dummy.jpg"))
                h, w = get_image_properties(first_path_as_sample)
                create_blank_dummy(h, w, dummy_path)
                dummy_path_list.append(dummy_path)
        else:
            dummy_name = image_name.replace(old_subj_id, ids_tup[0]).replace(old_area_id, ids_tup[1]).replace(old_time_id, ids_tup[2])
            dummy_path = first_path_as_sample.replace(image_name, dummy_name.replace(".jpg", "_dummy.jpg"))
            h, w = get_image_properties(first_path_as_sample)
            create_blank_dummy(h, w, dummy_path)
            dummy_path_list.append(dummy_path)

    return dummy_path_list

def create_code_dict_for_excel_table(path):
    file_extension = "*.jpg"

    filenamelist = Util.getAllFiles(path, file_extension, depth=-1)
    area = re.compile(r'F[0-9]{2}')
    time = re.compile(r'T[0-9]{2}')

    areas_set = set([area.search(fname).group(0) for fname in filenamelist ])
    times_set = set([time.search(fname).group(0) for fname in filenamelist ])

    area_dict = {a : f"$area name$${a}$" for a in areas_set}
    time_dict = {t : f"$time name$${t}$" for t in times_set}
    area_dict.update(time_dict)
    # merging time dict into area dict and then returning the merged area dict
    return area_dict

# Use for testing above functions only; else comment this part out

# if __name__ == "__main__":
#     from ImageAnalysis import Util
#     PATH = r"D:\Code\Software_test_sample_data\Dev_Proj_5__ExcelToPdfConverter\SAMPLE_DATAPOINTS\22.0229-39_wrink_USRCLIP"
#     filenamelist = Util.getAllFiles(PATH, "*.jpg", depth=-1)
#     dict_code = create_code_dict_for_excel_table(filenamelist)
#     print(dict_code)
   