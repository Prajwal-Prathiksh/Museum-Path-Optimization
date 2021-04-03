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
L = [i.split('.')[0] for i in L]
# print(L)

prefix = os.path.join(
    os.getcwd().split(keyw)[0],
    keyw,
    'data',
    'TSPLIB',
    'Asymmetric',
    'Opt')

prefix2 = os.path.join(
    os.getcwd().split(keyw)[0],
    keyw,
    'data',
    'Asym_TSPLIB_opt_costs.txt')

f = open(prefix2, 'r')
data = f.readlines()
f.close()
datL = {i.split()[0]: i.split()[-1] for i in data}
# print(datL)

for i in datL.keys():
    path = os.path.join(prefix, i.split('.')[0] + '.opt.tour.txt')
    print(path)
    f = open(path, 'w')
    f.write('NAME : ' + i + '\n')
    f.write('COMMENT : ' + '\n')
    f.write('TYPE : ' + '\n')
    f.write('DIMENSION : ' + '\n')
    f.write('TOUR SECTION' + '\n')
    f.write('EOF' + '\n')
    f.write(datL[i])
    f.close()
