f = open("tt.txt", "r")
w = open("slicer_opt_tt.list", "w")
for line in f:
    line = line.strip()
    line = line.replace('/data','/media/qc/1')
    print(line)
    w.write(line+"\n")
