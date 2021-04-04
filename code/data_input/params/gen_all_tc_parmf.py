import os

keyw = 'Museum-Path-Optimization'
prefix = os.path.join(
    os.getcwd().split(keyw)[0],
    keyw,
    'data',
    'TSPLIB',
    'Symmetric',
    'XML')

L = sorted(os.listdir(prefix))
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
    'All_TC_Sym_XML.txt')
f = open(fname, 'w')
f.write('SYM_TSPLIB_XML\n')
f.write('SYM\n')
f.writelines(L)
f.close()
