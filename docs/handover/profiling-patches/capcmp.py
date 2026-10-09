import struct, sys, math
def load(f):
    d={}; order=[]; b=open(f,"rb").read(); i=0
    while i<len(b):
        n=struct.unpack_from("i",b,i)[0]; i+=4; name=b[i:i+n].decode(); i+=n
        m=struct.unpack_from("i",b,i)[0]; i+=4; v=struct.unpack_from("%df"%m,b,i); i+=4*m
        d[name]=v; order.append(name)
    return d, order
a,oa=load(sys.argv[1]); b,_=load(sys.argv[2])
first=None
for n in oa:
    if n not in b: continue
    md=max(abs(x-y) for x,y in zip(a[n],b[n])); rel=md/(max(abs(x) for x in a[n])+1e-30)
    if md>0 and first is None: first=n
    print(f"{n:<20} max|d|={md:.3g} rel={rel:.3g}")
print("first differing:", first)
