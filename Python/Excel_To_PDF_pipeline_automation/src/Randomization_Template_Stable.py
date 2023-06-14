from Common_Functions_Stable import validate_random_file, extract_elements_from_regex_mask

def randomize_std(filelist, randomfilepath, area_id_list, time_id_list, sorted_subject_id_list):
    """
    randomize for standard template
    """ 
    # create tuple (filepath, (subid, *areaid/timeid)) # *needs to be changed depending on template requirement
        # regex to check if ids from iterator match ids in filenames - exception when not matching

    tuple_list_with_path_and_ids = []

    for subj in sorted_subject_id_list:
        for ar in area_id_list:
            for tim in time_id_list:
                for filepath in filelist:
                    filenm = filepath.split("\\")[-1]
                    if subj in filenm and ar in filenm and tim in filenm:
                        # validation
                        val_code = extract_elements_from_regex_mask(filenm, r"S[0-9]{3}F[0-9]{2}")
                        comparison = str(subj + ar)
                        if  val_code == comparison:
                            tuple_list_with_path_and_ids.append((filepath, (subj, ar)))
                        else:
                            print(f"val id in path({val_code}) does not match with comparision iterator ({comparison})")
                            raise BaseException

    print(tuple_list_with_path_and_ids)  #[('D:\\STUDIES\\23.0058-6...ntour1.jpg', ('S001', 'F01')), ('D:\\STUDIES\\23.0058-6...ntour2.jpg', ('S001', 'F02'))]

    # read random sequence from the randomization file (.txt file in study folder) and validate subject ids (check if file has equal number of subjects)
    cleaned_list = validate_random_file(randomfilepath, sorted_subject_id_list) # only seq of 4 letters per line
    print(cleaned_list) # ['AB', 'BA', 'BA', 'AB', 'BA'...]

    # create a dictionary with tuple (sub, arid) as key and value as Product alphabet from line per subject 
    # { (S001, F01): B, (S001, F02): C, (S001, F03): D, (S001, F04): A, etc}

    mapping_dict = {}
    for line, subj in zip(cleaned_list, sorted_subject_id_list):
        for p_id, ar in zip(line, area_id_list):
            mapping_dict[(subj, ar)] = p_id        
    
    print(mapping_dict) #{('S001', 'F01'): 'A', ('S001', 'F02'): 'B', ('S002', 'F01'): 'B',...}

    # map each tuple in tuple_list_with_ids with ids(values) using mapping_dict's keys with tuple idicies(tuple[0])

    derandomized_list = []
    for subj in sorted_subject_id_list:
        grouped_by_subj = [(tupl[0], mapping_dict[tupl[1]]) for tupl in tuple_list_with_path_and_ids if subj==tupl[1][0]]
        derandomized_paths_grouped_per_subj = sorted(grouped_by_subj, key=lambda x: x[1])
        derandomized_list.append(derandomized_paths_grouped_per_subj) # [[('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'A'), ('D:\\STUDIES\\23.0058-6...ntour2.jpg', 'A'), ('D:\\STUDIES\\23.0058-6...ntour2.jpg', 'A'), ('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'B'), ('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'B'), ('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'B')], [(...), (...), (...), (...), (...), (...)]
    
    final_derandomized_list = []
    for sub_list in derandomized_list:
        for tup in sub_list:
            final_derandomized_list.append(tup)

    print(final_derandomized_list) # [('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'A'), ('D:\\STUDIES\\23.0058-6...ntour1.jpg', 'A'), (path, 'A'),(path, 'B'),(path, 'B'), (path, 'B')...(...),(...),(...)]

    final_randomised_list_iter = iter(final_derandomized_list)
    return final_randomised_list_iter