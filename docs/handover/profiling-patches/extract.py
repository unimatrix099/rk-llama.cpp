import re, json, sys
cycles=[]; cur=[]
for l in open(sys.argv[1]):
    m=re.search(r"draft candidate   0, pos +(\d+): +(\d+)", l)
    if m: cur.append(int(m.group(2))); continue
    m=re.search(r"generate_draft: id=(\d+), #tokens=(\d+), #draft=(\d+), pos_next=(\d+)", l)
    if m: cycles.append({"id_last":int(m.group(1)),"pos":int(m.group(4)),"draft":cur[-int(m.group(3)):] if int(m.group(3)) else []}); cur=[]; continue
    m=re.search(r"add accepted tokens: sampled=(\d+), ids.size=(\d+)", l)
    if m and cycles: cycles[-1]["sampled"]=int(m.group(1)); cycles[-1]["n_acc"]=int(m.group(2))-1
json.dump(cycles, open(sys.argv[2],"w")); print(len(cycles), "cycles; first", cycles[0])
