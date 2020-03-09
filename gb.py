import os

path='./data/chart_data'
file_list=os.listdir(path)
file_list_csv=[file for file in file_list if file.endswith(".csv")]

print(file_list_csv)