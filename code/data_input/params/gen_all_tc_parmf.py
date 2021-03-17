import os

keyw = 'Museum-Path-Optimization'
prefix = os.path.join(
    os.getcwd().split(keyw)[0],
     keyw, 
     'data', 
     'TSPLIB', 
     'Symmetric',
     'XML')

L = os.listdir(prefix)
L.sort()
L = [i.split('.')[0] + '\n' for i in L]

f = open('AllTC_Sym_XML', 'w')
f.write('TSPLIB_XML\n')
f.write('SYM\n')
f.writelines(L)

