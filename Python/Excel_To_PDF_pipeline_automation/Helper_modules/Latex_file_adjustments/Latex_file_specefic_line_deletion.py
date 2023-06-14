import re
import os

PATH_TO_ORIG_FILE = r"D:\STUDIES\22.0340-51_scratch_woundheal\piased\reg_new\cropped_datspc9_pias\cropped_again\jpg_converted\jpg_small50\Test_results\1.tex"

rootPath = "\\".join(PATH_TO_ORIG_FILE.split("\\")[:-1])
newLatexFileName = PATH_TO_ORIG_FILE.split("\\")[-1].replace(".tex","_updated.tex")
PATH_TO_NEW_FILE = os.path.join(rootPath, newLatexFileName)
# COLUMN_INDEX_TO_POP = 1 # 2rd product column containing #RR 4001
COLUMN_INDEX_TO_POP = 2 # 3rd product column containing RR 5001


class EmptyObjectReturnedError(Exception):
    pass

def search_text(mask_pattern, line_to_search):
    try:
        return re.search(mask_pattern, line_to_search)
    except AttributeError:
        return None


def Delete_specific_lines(path_to_latex_file, path_to_new_latex_file, column_idx_to_pop):

    with open(path_to_latex_file, "r") as latexF:
        LFlines = latexF.readlines()

    with open(path_to_new_latex_file, "w") as NewLatexF:
        for line in LFlines:

            if "&&" in line:
                line_list = line.split("&&")
                line_list.pop(column_idx_to_pop)
                mod_line = "&&".join(line_list)
                NewLatexF.write(mod_line)
                continue
            else:
                NewLatexF.write(line)

def main():
    Delete_specific_lines(PATH_TO_ORIG_FILE, PATH_TO_NEW_FILE, COLUMN_INDEX_TO_POP)

if __name__ == "__main__":
    main()
    input("Press Enter to continue...")