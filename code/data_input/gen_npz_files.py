###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
import os
import numpy as np

from code.data_input.input_final import get_input_loader


###########################################################################
# Code
###########################################################################


def gen_npz_from_xml(all_tc_fname, ds_type):
    bl = get_input_loader(all_tc_fname, False)
    ds_map = {'SYM': 'Symmetric', 'ASYM': 'Asymmetric'}
    keyw = 'Museum-Path-Optimization'
    pefix = os.path.join(
        os.getcwd().split(keyw)[0],
        keyw, 'data', 'TSPLIB',
        ds_map[ds_type], 'NPZ',
    )
    for ip in bl.get_input_test_cases().values():
        fname = os.path.join(pefix, ip.get_dataset_name()) + '.npz'
        name = ip.get_dataset_name()
        cost_m = ip.get_cost_matrix()
        np.savez(
            fname,
            NAME=name,
            EDGE_WEIGHT_SECTION=cost_m
        )


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    gen_npz_from_xml('All_TC_Sym_XML.txt', 'SYM')
