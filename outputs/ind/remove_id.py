import re
pat = re.compile(r"^id:\d+")


with open("rst3.txt") as f_r, open("rst3_.txt", "w") as f_w:
    lines = f_r.readlines()

    for line in lines:
        if pat.search(line):
            line = line.replace("\tNNP", "")

        f_w.write(line)
