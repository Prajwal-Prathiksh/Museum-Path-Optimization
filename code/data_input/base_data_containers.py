###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
import os
import numpy as np

###########################################################################
# Code
###########################################################################


class BaseLibraryFileContainer:
    def __init__(self, file_path=None):
        self.file_path_name = file_path
        self.dataset_name = ''
        self.comment = ''

    def get_dataset_name(self):
        return self.dataset_name


class BaseTestCasePSContainer(BaseLibraryFileContainer):
    def __init__(self, file_path=None):
        BaseLibraryFileContainer.__init__(self, file_path)
        self.cost_data_container = {}
        self.constraint_data_container = {}
        self.additional_data_container = {}

    def update_cost_data(self):
        pass

    def update_constraint_data(self):
        pass

    def get_cost_data(self):
        return self.cost_data_container

    def get_constraint_data(self):
        return self.constraint_data_container

    def get_additional_data(self):
        return self.additional_data_container


class BaseOptimalDataContainer(BaseLibraryFileContainer):
    def __init__(self, file_path=None):
        BaseLibraryFileContainer.__init__(self, file_path)
        self.optimal_route_data = {'route': [], 'cost': None}

    def get_opt_data(self):
        return self.optimal_route_data


class BaseTspFileContainer(BaseTestCasePSContainer):
    def __init__(self, file_path=None):
        BaseTestCasePSContainer.__init__(self, file_path)
        self.dimension = 0
        self.type = ''


class BaseTsplibFileContainer(BaseTspFileContainer):
    def __init__(self, file_path=None):
        BaseTspFileContainer.__init__(self, file_path)
        self.cost_matrix = np.array([])

    def update_cost_data(self):
        self.cost_data_container['cost_matrix'] = self.cost_matrix


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    # ob = XMLFileRead('a280.xml')
    pass
