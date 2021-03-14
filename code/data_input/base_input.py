from data_containers.TSPLIB_data_container import TsplibXMLFileContainer

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
        BaseInputData.__init__(self,file_read_cont)

    def get_cost_matrix(self):
        return self.cost_data['cost_matrix']

class BaseInputLoader:
    def __init__(self, load_list_path):
        self.input_class = BaseInputData
        self.file_read_type = TsplibXMLFileContainer
        self.file_type_map = {'TSPLIB_XML':[TsplibXMLFileContainer, TSPLIBCostMatrixInput]}
        self.file_reader_list = []
        self.input_test_cases = {}

        self.read_tc_paths(load_list_path)
        self.generate_input_test_cases()

    def update_file_type(self, file_type):
        if file_type in list(self.file_type_map.keys()):
            self.file_read_type = self.file_type_map[file_type][0]
            self.input_class = self.file_type_map[file_type][1]

    def update_input_class(input_class):
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
            self.input_test_cases['TEST_CASE_'+str(i+1)] = self.input_class(self.file_reader_list[i])

    def get_input_test_cases(self):
        return self.input_test_cases

    def get_input_test_case(self, test_case_no):
        key = 'TEST_CASE_'+str(test_case_no)
        if key in self.input_test_cases:
            return self.input_test_cases[key]
        return None


loader = BaseInputLoader('data_input/test_load_list')
print(loader.get_input_test_cases())
print(loader.get_input_test_case(1).get_cost_matrix())