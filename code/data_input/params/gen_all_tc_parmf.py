import os

keyw = 'Museum-Path-Optimization'
prefix = os.path.join(
    os.getcwd().split(keyw)[0],
     keyw, 
     'data', 
     'TSPLIB', 
     'Asymmetric',
     'XML')

L = os.listdir(prefix)
L.sort()
L = [i.split('.')[0] + '\n' for i in L]
print(L)

prefix = os.path.join(
    os.getcwd().split(keyw)[0],
     keyw)
fname = os.path.join(
    prefix,
    'code',
    'data_input',
    'params',
    'AllTC_Asym_XML')
f = open(fname, 'w')
f.write('ASYM_TSPLIB_XML\n')
f.write('ASYM\n')
f.writelines(L)
f.close()

