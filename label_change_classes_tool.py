"""

MIT License

Copyright 2024 Maximilian Petersson and Nahom Solomon

"""



import os

def process_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith('0'):
            lines[i] = '0' + lines[i][1:]
        i+=1

    with open(file_path, 'w') as file:
        file.writelines(lines)

def change_class_label(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            process_text_file(file_path)


#main
folder_path = ''

change_class_label(folder_path)
