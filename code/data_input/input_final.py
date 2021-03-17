###########################################################################
# Imports
###########################################################################
# Standard library imports
import os
import numpy as np

from bass_input import BaseInputLoader
from file_reader import TxtFileRead
###########################################################################
# Code
###########################################################################


class GenInternalParams(TxtFileRead):
    def __init__(self, param_file_path):
        self.file_type_map = {'SYM_TSPLIB_XML': [
            'XML', '.xml'], 'TSPLIB_NPZ': ['NPZ', '.npz'],
            'ASYM_TSPLIB_XML': ['XML', '.xml']}
        self.dataset_type_map = {'SYM': 'Symmetric', 'ASYM': 'Asymmetric'}
        self.param_path = param_file_path
        self.int_param_paths = []
        TxtFileRead.__init__(self, param_file_path)

    def process_file_data(self):
        file_type = self.file_data[0][:-1]
        ds_type = self.file_data[1][:-1]
        names = [i[:-1] for i in self.file_data[2:]]
        # Might have to change
        keyw = 'Museum-Path-Optimization'
        pefix = os.path.join(os.getcwd().split(keyw)[0], keyw)
        ###

        data_pefix = os.path.join(pefix, 'data', 'TSPLIB',
                                  self.dataset_type_map[ds_type])
        paths1 = [os.path.join(data_pefix, self.file_type_map[file_type][0], i) +
                  self.file_type_map[file_type][1] + '\n' for i in names]
        paths2 = [os.path.join(data_pefix, 'Opt', i) +
                  '.opt.tour.txt' + '\n' for i in names]

        internal_nam = self.param_path.split('.')[0].split('/')[-1]
        int_pefix = os.path.join(pefix, 'code', 'data_input', 'internal_params',
                                 self.dataset_type_map[ds_type], internal_nam)
        int_fnames = [int_pefix + '_tc', int_pefix + '_opt']
        self.int_param_paths = int_fnames

        write_to_file(int_fnames[0], [file_type + '\n'] + paths1)
        write_to_file(int_fnames[1], paths2)

    def get_int_param_paths(self):
        return self.int_param_paths


def write_to_file(fpath, lines):
    f = open(fpath, 'w')
    f.writelines(lines)
    f.close()


def get_input_loader(param_file_name, load_opt_solns=True):
    keyw = 'Museum-Path-Optimization'
    prefix = os.path.join(os.getcwd().split(keyw)[0], keyw, 'code',
                          'data_input', 'params', param_file_name)

    g = GenInternalParams(prefix)
    tc_path, opt_path = g.get_int_param_paths()
    if not load_opt_solns:
        opt_path = None
    return BaseInputLoader(tc_path, opt_path)


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    bl = get_input_loader('Choose_TC_Asym_NPZ.txt', False)
    print(bl.get_input_test_cases())
    print(bl.get_input_test_case(1))
    print(bl.get_input_test_case(1).get_cost_matrix())
