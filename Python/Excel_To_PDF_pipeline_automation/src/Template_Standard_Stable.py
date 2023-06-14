from Internal_Imports_Stable import studynumber, Test_type
from Internal_Imports_Stable import header, header_draft, pagestyle, hypersetup
from Latex_File_Create_Stable import create_final_latex_file
from Common_Functions_Stable import create_Latex_document

def standardize(area_code_list, time_code_list, area_sorted, time_sorted, sub_sorted, filepaths, filenamelist, rownames, colnames, draft_flag, random_iter):
    """
    Standard template - User needs to only change the column and row names
    """    
    nr_col=len(time_code_list)#len(time_list)
    nr_row=len(area_code_list)#len(area_list)

    max_height=round(1.0/(1.25*nr_row), 3) 
    max_width=round(1.0/(1.5*nr_col), 3) 

    # lax_document, new_page_alert = create_Latex_document(colnames, rownames, sub_sorted, area_sorted, time_sorted, filepaths, filenamelist, max_height, max_width) 
    lax_document, new_page_alert = create_Latex_document(colnames, rownames,  sub_sorted, max_height, max_width, area_sorted, time_sorted, filepaths, filenamelist, Type="Standard", RandomList=random_iter)

    std_tex_file = create_final_latex_file(studynumber, header, header_draft, pagestyle, hypersetup, lax_document, new_page_alert, colnames, Test_type, draft=draft_flag)

    return std_tex_file

