###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
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
        self.OUTPUT_DIR = os.path.join(os.getcwd(), 'data', 'cost_matrices')

        self.input_class = BaseInputData
        self.file_read_type = TsplibXMLFileContainer
        self.file_type_map = {'TSPLIB_XML': [
            TsplibXMLFileContainer, TSPLIBCostMatrixInput]}
        self.file_reader_list = []
        self.input_test_cases = {}
        self.read_tc_paths(load_list_path)

        self.generate_input_test_cases()
        self.store_cost_matrices()

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

    def get_test_case_number(self, test_case_name):
        N = self.get_number_of_test_cases()
        for idx in range(1, N + 1):
            if test_case_name == self.get_test_case_name(idx):
                return idx

    def get_number_of_test_cases(self,):
        return len(self.get_input_test_cases())

    def store_cost_matrices(self, quiet=False):
        '''
            Stores the cost matrices of all the test cases as `.npz` files.

            NOTE:
                This runs only if
                    `./data/cost_matrices/`
                directory does not exist!

            Parameters:
            -----------
            quiet: (Boolean), default=False
                If True, prints output prompts.
        '''
        OUTPUT_DIR = self.OUTPUT_DIR

        if os.path.exists(OUTPUT_DIR) is False:
            os.mkdir(OUTPUT_DIR)

            N = self.get_number_of_test_cases()

            for i in range(1, N + 1):
                cost_matrix = self.get_input_test_case(i).get_cost_matrix()
                test_case_name = self.get_test_case_name(i)
                fname = os.path.join(OUTPUT_DIR, f'{test_case_name}.npz')

                if not quiet:
                    print('=================================================')
                    print(test_case_name)
                    print(cost_matrix)
                    print('=================================================')

                np.savez(
                    fname,
                    test_case_name=test_case_name,
                    cost_matrix=cost_matrix)
            if not quiet:
                print('Done: Store all cost matrices.')


class TestCaseLoader:
    '''
        TestCaseLoader
        --------------

        To load all the test cases from the
                `./data/cost_matrices/`
        subdirectory.

        Class Variables:
        ----------------
        OUTPUT_DIR: (string)
            The absolute path of `./data/cost_matrices/` on one's PC
        test_cases_data: (List)
            All the test case data, stored as `numpy.lib.npyio.NpzFile`
            instances
        num_test_cases: (int)
            Total number of test cases which have been loaded
        names_test_cases: (List)
            The names of all the test cases

        Class Methods:
        --------------
        get_test_case_number():
            To obtain the test case number for a given test case name
        get_test_data():
            To obtain the test name and the cost matrix for a given test case
            identifier
    '''

    def __init__(self):
        self.OUTPUT_DIR = os.path.join(os.getcwd(), 'data', 'cost_matrices')
        self.test_cases_data = self.__load_test_cases()
        self.num_test_cases = len(self.test_cases_data)

        tc_names = []
        for idx in range(self.num_test_cases):
            data = self.test_cases_data
            tc_names.append(str(data[idx]['test_case_name']))

        self.names_test_cases = tc_names

    # Private methods
    def __load_test_cases(self):
        '''
            Load all the test cases from the
                `./data/cost_matrices/`
            subdirectory.

            Returns:
            --------
            data: (List)
                All the test case data, stored as `numpy.lib.npyio.NpzFile`
                instances
        '''
        OUTPUT_DIR = self.OUTPUT_DIR
        data = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".npz"):
                fpath = os.path.join(OUTPUT_DIR, filename)
                data.append(np.load(fpath))
        return data

    # Public methods
    def get_test_case_number(self, tc_name):
        '''
            Obtain the test case number for a given test case name.

            Parameters:
            -----------
            tc_name: (string)
                Test case name

            Returns:
            --------
            tc_number: (int)
                Test case number
        '''
        N = self.num_test_cases
        tc_number = []
        for idx in range(N):
            if tc_name == self.names_test_cases[idx]:
                tc_number.append(idx)
                break

        if not tc_number:
            raise Exception('KeyError: Invalid test case name')
        else:
            return tc_number[0]

    def get_test_data(self, case_identifier):
        '''
            Obtain the test name and the cost matrix for a given test case
            identifier.

            Parameters:
            -----------
            case_identifier: (int/string)
                Can refer to either the test case number, whose range is
                (0, self.num_test_cases) or can  refer to the test case name

            Returns:
            --------
            tc_name: (string)
                Name of the test case
            cost_matrix: (matrix)
                Cost matrix
        '''
        if isinstance(case_identifier, str):
            tc_number = self.get_test_case_number(case_identifier)
        elif isinstance(case_identifier, int):
            tc_number = case_identifier
            if tc_number >= self.num_test_cases or tc_number < 0:
                msg = f'Enter value between [0, {self.num_test_cases-1}].'
                raise Exception(f'KeyError: Invalid test number. {msg}')

        data = self.test_cases_data[tc_number]

        tc_name = str(data['test_case_name'])
        cost_matrix = data['cost_matrix']

        return tc_name, cost_matrix


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    fpath = os.path.join(os.getcwd(), 'code', 'data_input', 'test_load_list')
    loader = BaseInputLoader(fpath)
