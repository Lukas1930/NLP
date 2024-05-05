import re

with open('StarWars_Raw_Sentences.csv', 'r', encoding="utf-8") as file:
    with open('StarWars_Sentences.txt', 'w', encoding="utf-8") as newfile:
        for line in file:
            new_line = re.sub(r'^\d+,', '', line)
            new_line = new_line.replace('"', '')
            new_line.strip()
            if new_line != "\n":
                newfile.write(new_line)