with open("train_urls.txt", "r") as f:
    lines = [line.strip("\n") for line in f.readlines()]

lines = [line[:-5] + "w.jpg" for line in lines]

with open("train_urls.txt", "w") as f:
    for line in lines:
        line += "\n"
        f.write(line)