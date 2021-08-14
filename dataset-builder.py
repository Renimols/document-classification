import csv
import os
import re

directory = '/home/archana/Desktop/reni/bbc/'
with open("/home/archana/Desktop/test_data.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["documents", "categories"])
    for foldername in os.listdir(directory):
        for filename in os.scandir(''.join(directory+foldername)):
            with open(filename.path, encoding="utf8", errors='ignore') as file:
                writer.writerow((file.readlines(), foldername))
