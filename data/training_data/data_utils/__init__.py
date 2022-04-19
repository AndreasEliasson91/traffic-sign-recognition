from os import listdir, rename
from os.path import isfile, join

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'P', 'S', 'T', 'X', 'Z']


def set_annotations(file_dir: str, out_dir: str) -> None:
    with open(file_dir, 'w') as out_file:
        for i in range(len(LETTERS)):
            for file in listdir(out_dir):
                if isfile(join(out_dir, file)) and LETTERS[i] in file:
                    out_file.write(f'{file},{LETTERS[i]}\n')


def change_img_names(in_dir: str, out_dir: str) -> None:
    for i in range(len(LETTERS)):
        for j, file in enumerate(listdir(in_dir)):
            if LETTERS[i] in file:
                rename(f'{in_dir}{file}', f'{out_dir}{LETTERS[i]}_0{j+1}.jpg')


# change_img_names('../', '../signs/')
set_annotations('../signs/annotation/training_annotations.csv', '../signs/')
