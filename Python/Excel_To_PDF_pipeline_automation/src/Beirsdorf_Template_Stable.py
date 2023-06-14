import re
import os

def return_correct_digit(orig_id):
    if orig_id[0] != "0":
        return orig_id
    else:
        return orig_id[-1]


def do_the_Beirsdorf_transform(fpath, beir_rand_fpath):
    """
    Changes the latex page headers (Subject Ids in particular) to the Beiersdorf randomization file
    """    
    # From Beirsdorf_random text file extract dict --> 
    # {1:"20/Cleansner A + Fluid A(morning) + Fluid B(evening) ",  2: "Cleansner B + Fluid B(morning) + Fluid B(evening)"}...
    Id_mask = re.compile(r"Subject [0-9]*")

    # mask for getting the subject number in "l" element in lines_gen
    rand_dict = {}

    with open(beir_rand_fpath) as frand:
        rand_gen = iter(frand.readlines())

    while True:
        try:
            l = next(rand_gen)
            rand_dict[l.split("\t")[0].replace("ï»¿", "")] = l.split("\t")[1] # stupid "ï»¿" character magically but annoyingly appears so should be forecefully replaced
        except StopIteration:
            break
    print(rand_dict)


    with open(fpath) as ftex:
        lines_gen = iter(ftex.readlines())
    print(lines_gen)
    tex_temp = []
    while True:
        try:
            l = next(lines_gen)
            if "Subject" in l:
                l_interest = l
                # intersting element
                Subj_id = Id_mask.search(l).group(0)
                print(Subj_id[-2:])
                dict_returned = rand_dict[return_correct_digit(Subj_id[-2:])].replace("\n", "")
                if dict_returned == "30":
                    Prod = "Cleansner A + Fluid A(morning) + Fluid A(evening)"
                elif dict_returned == "20":
                    Prod = "Cleansner B + Fluid B(morning) + Fluid B(evening)"
                l_mod = l_interest.replace(f"{Subj_id}", f"{Subj_id} ({Prod})")
                # regex search "l" and get the subject id
                # l.replace(subjid, "subid + dict[int(subject id)]")
                tex_temp.append(l_mod)
            else:
                tex_temp.append(l)
        except StopIteration:
            break

            
    new_tex_gen = iter(tex_temp)

    new_file_name = fpath.split("\\")[-1][:-4]
    with open(os.path.join(os.getcwd(),f"{new_file_name}_appended.tex"), "w") as f_new:
        while True:
            try:
                f_new.write(next(new_tex_gen))
            except StopIteration:
                break


    return new_file_name, f_new.name


