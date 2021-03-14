###########################################################################
# Imports
###########################################################################
# Standard library imports
import os
from xml.dom import minidom
import numpy as np

###########################################################################
# Code
###########################################################################


class FileReadContainer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_data = []
        self.data_container = []
        self.read_file()
        self.process_file_data()

    def read_file(self):
        pass

    def process_file_data(self):
        pass


class XMLFileRead(FileReadContainer):
    def __init__(self, file_path):
        FileReadContainer.__init__(self, file_path)

    def read_file(self):
        self.file_data = minidom.parse(self.file_path)


class BaseLibraryFileContainer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cost_data_container = {}
        self.constraint_data_container = {}
        self.dataset_name = ''
        self.comment = ''

    def update_cost_data(self):
        pass

    def update_constraint_data(self):
        pass

    def get_cost_data(self):
        return self.cost_data_container

    def get_constraint_data(self):
        return self.constraint_data_container

    def get_dataset_name(self):
        return self.dataset_name


class BaseTspFileContainer(BaseLibraryFileContainer):
    def __init__(self, file_path):
        BaseLibraryFileContainer.__init__(self, file_path)
        self.dimension = 0
        self.type = ''


class BaseTsplibFileContainer(BaseTspFileContainer):
    def __init__(self, file_path):
        BaseTspFileContainer.__init__(self, file_path)
        self.cost_matrix = np.array([])


class TsplibXMLFileContainer(BaseTsplibFileContainer, XMLFileRead):
    def __init__(self, file_path):
        BaseTsplibFileContainer.__init__(self, file_path)
        XMLFileRead.__init__(self, file_path)

    def process_file_data(self):
        self.dataset_name = self.file_data.getElementsByTagName('name')[
            0].firstChild.data
        self.comment = self.file_data.getElementsByTagName('description')[
            0].firstChild.data
        vertices = self.file_data.getElementsByTagName('vertex')
        self.dimension = len(vertices)
        self.cost_matrix = np.zeros([self.dimension, self.dimension])
        for vert_indx in range(len(vertices)):
            for i in range(1, len(vertices[vert_indx].childNodes), 2):
                self.cost_matrix[
                    vert_indx, (i + ((i // 2) >= (vert_indx))) // 2
                ] = float(
                    vertices[vert_indx].childNodes[i]
                    ._get_attributes().items()[0][1]
                )

        self.update_cost_data()

    def update_cost_data(self):
        self.cost_data_container['cost_matrix'] = self.cost_matrix


class BaseInputData:
    def __init__(self, file_read_cont):
        self.dataset_name = ''
        self.cost_data = {}
        self.constraint_data = {}
        self.generate_input(file_read_cont)

    def generate_input(self, file_read_cont):
        self.dataset_name = file_read_cont.get_dataset_name()
        self.cost_data = dict(file_read_cont.get_cost_data())
        self.constraint_data = dict(file_read_cont.get_constraint_data())

    def get_dataset_name(self):
        return self.dataset_name


class TSPLIBCostMatrixInput(BaseInputData):
    def __init__(self, file_read_cont):
        BaseInputData.__init__(self, file_read_cont)

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

    def __init__(self, load_list_path):
        self.input_class = BaseInputData
        self.file_read_type = TsplibXMLFileContainer
        self.file_type_map = {'TSPLIB_XML': [
            TsplibXMLFileContainer, TSPLIBCostMatrixInput]}
        self.file_reader_list = []
        self.input_test_cases = {}

        self.read_tc_paths(load_list_path)
        self.generate_input_test_cases()

    def update_file_type(self, file_type):
        if file_type in list(self.file_type_map.keys()):
            self.file_read_type = self.file_type_map[file_type][0]
            self.input_class = self.file_type_map[file_type][1]

    def update_input_class(self, input_class):
        if isinstance(input_class, BaseInputData):
            self.input_class = input_class

    def read_tc_paths(self, load_list_path):
        f = open(load_list_path, 'r')
        file_type = f.readline()[:-1]
        self.update_file_type(file_type)
        read_paths = f.readlines()
        for path in read_paths:
            self.file_reader_list.append(self.file_read_type(path[:-1]))
        f.close()

    def generate_input_test_cases(self):
        for i in range(len(self.file_reader_list)):
            self.input_test_cases['TEST_CASE_' + str(i + 1)] = \
                self.input_class(self.file_reader_list[i])

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


if __name__ == '__main__':
    fpath = os.path.join(os.getcwd(), 'code', 'data_input', 'test_load_list')
    loader = BaseInputLoader(fpath)
    print('\n\nTest Cases:')
    print('=================================================================')
    print(loader.get_input_test_cases())
    print('\n\nCost Matrices:')
    print('=================================================================')
    for i in range(len(loader.get_input_test_cases())):
        print(f'<<<< Cost Matrix - {i+1} >>>>')
        print(loader.get_input_test_case(i + 1).get_cost_matrix())
        print('\n')
