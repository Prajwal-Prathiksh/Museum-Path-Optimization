###########################################################################
# Imports
###########################################################################
# Standard library imports
import os
import numpy as np

from data_containers import TsplibXMLFileContainer, TsplibNpzFileContainer, TsplibOptDataContainer
###########################################################################
# Code
###########################################################################


class BaseInputData:
    def __init__(self, file_read_cont=None, opt_file_cont=None):
        self.dataset_name = ''
        self.cost_data = {}
        self.constraint_data = {}
        self.optimal_data = {'route': [], 'cost': None}
        self.generate_input(file_read_cont, opt_file_cont)

    def generate_input(self, file_read_cont, opt_file_cont):
        self.dataset_name = file_read_cont.get_dataset_name()
        if file_read_cont != None:
            self.cost_data = dict(file_read_cont.get_cost_data())
            self.constraint_data = dict(file_read_cont.get_constraint_data())
        if opt_file_cont != None:
            self.optimal_data = dict(opt_file_cont.get_opt_data())

    def get_dataset_name(self):
        return self.dataset_name

    def get_cost_data(self):
        return self.cost_data

    def get_constraint_data(self):
        return self.constraint_data

    def get_opt_data(self):
        return self.optimal_data

    def get_opt_route(self):
        return self.optimal_data['route']

    def get_opt_cost(self):
        return self.optimal_data['cost']


class TSPLIBCostMatrixInput(BaseInputData):
    def __init__(self, file_read_cont, opt_file_cont):
        BaseInputData.__init__(self, file_read_cont, opt_file_cont)

    def get_cost_matrix(self):
        return self.cost_data['cost_matrix']


class BaseInputLoader:
    '''
        BaseInputLoader Class
        ----------------
        To be used to load test cases from file.

        Parameters:
        -----------
        load_list_path: (str)
            path of file which conttains paths of all test case data files.
            First line of file should be 'TSPLIB_XML' for now. Rest of the
            lines should be paths to test cases
    '''

    def __init__(self, tc_load_list_path=None, opt_load_list_path=None):
        self.OUTPUT_DIR = os.path.join(os.getcwd(), 'data')

        self.input_class = BaseInputData
        self.file_read_type = TsplibXMLFileContainer
        self.opt_file_read_type = TsplibOptDataContainer
        self.file_type_map = {'TSPLIB_XML': [
            TsplibXMLFileContainer, TSPLIBCostMatrixInput, TsplibOptDataContainer],
            'TSPLIB_NPZ': [TsplibNpzFileContainer, TSPLIBCostMatrixInput,
                           TsplibOptDataContainer]}
        self.tc_file_reader_list = []
        self.opt_file_reader_list = []
        self.input_test_cases = {}
        if tc_load_list_path != None:
            self.read_tc_paths(tc_load_list_path)
        if opt_load_list_path != None:
            self.read_opt_paths(opt_load_list_path)

        self.generate_input_test_cases()

    def update_file_type(self, file_type):
        if file_type in list(self.file_type_map.keys()):
            self.file_read_type = self.file_type_map[file_type][0]
            self.input_class = self.file_type_map[file_type][1]
            self.opt_file_read_type = self.file_type_map[file_type][2]

    def update_input_class(self, input_class):
        if isinstance(input_class, BaseInputData):
            self.input_class = input_class

    def read_tc_paths(self, load_list_path):
        f = open(load_list_path, 'r')
        file_type = f.readline()[:-1]
        self.update_file_type(file_type)
        read_paths = f.readlines()
        for path in read_paths:
            self.tc_file_reader_list.append(self.file_read_type(path[:-1]))
        f.close()

    def read_opt_paths(self, opt_load_list_path):
        f = open(opt_load_list_path, 'r')
        read_paths = f.readlines()
        for path in read_paths:
            self.opt_file_reader_list.append(
                self.opt_file_read_type(path[:-1]))
        f.close()

    def generate_input_test_cases(self):
        # Hopefully Temp stuff
        compare_names = 1
        if len(self.tc_file_reader_list) != len(self.opt_file_reader_list):
            d = {-1: self.tc_file_reader_list, 1: self.opt_file_reader_list}
            keyd = np.sign(len(self.tc_file_reader_list) -
                           len(self.opt_file_reader_list))
            d[keyd] = [None for i in d[-keyd]]
            self.tc_file_reader_list = d[-1]
            self.opt_file_reader_list = d[1]
            compare_names = 0
        ###
        for i in range(len(self.tc_file_reader_list)):
            if compare_names:
                if self.tc_file_reader_list[i].get_dataset_name() == \
                        self.opt_file_reader_list[i].get_dataset_name():
                    self.input_test_cases['TEST_CASE_' + str(i + 1)] = \
                        self.input_class(self.tc_file_reader_list[i],
                                         self.opt_file_reader_list[i])
            else:
                self.input_test_cases['TEST_CASE_' + str(i + 1)] = \
                    self.input_class(self.tc_file_reader_list[i],
                                     self.opt_file_reader_list[i])

    def get_input_test_cases(self):
        return self.input_test_cases

    def get_input_test_case(self, test_case_no):
        key = 'TEST_CASE_' + str(test_case_no)
        if key in self.input_test_cases:
            return self.input_test_cases[key]
        return None

    def get_test_case_name(self, test_case_no):
        key = 'TEST_CASE_' + str(test_case_no)
        if key in self.input_test_cases:
            return self.input_test_cases[key].dataset_name
        return None

    def get_test_case_number(self, test_case_name):
        N = self.get_number_of_test_cases()
        for idx in range(1, N + 1):
            if test_case_name == self.get_test_case_name(idx):
                return idx

    def get_number_of_test_cases(self,):
        return len(self.get_input_test_cases())
