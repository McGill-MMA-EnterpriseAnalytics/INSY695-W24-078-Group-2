with open("/input/input_file.txt", "r", encoding = "utf-8") as in_f:

    with open("/data/first_output.txt", "w", encoding = "utf-8") as out_f:

        for line in in_f:

            out_f.write(line)
