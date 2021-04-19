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
from code.data_input.file_reader import XMLFileRead, NpzFileRead, TxtFileRead
from code.data_input.base_data_containers import (
    BaseTsplibFileContainer, BaseOptimalDataContainer
)


class SymTsplibXMLFileContainer(BaseTsplibFileContainer, XMLFileRead):
    def __init__(self, file_path=None):
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


class AsymTsplibXMLFileContainer(BaseTsplibFileContainer, XMLFileRead):
    def __init__(self, file_path=None):
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
                    vert_indx, (i - 1) // 2
                ] = float(
                    vertices[vert_indx].childNodes[i]
                    ._get_attributes().items()[0][1]
                )

        self.update_cost_data()


class TsplibNpzFileContainer(BaseTsplibFileContainer, NpzFileRead):
    def __init__(self, file_path=None):
        BaseTsplibFileContainer.__init__(self, file_path)
        NpzFileRead.__init__(self, file_path)

    def process_file_data(self):
        self.dataset_name = self.file_data['NAME']
        self.cost_matrix = self.file_data['EDGE_WEIGHT_SECTION']
        self.dimension = len(self.cost_matrix)

        self.update_cost_data()


class TsplibTxtFileContainer(BaseTsplibFileContainer, TxtFileRead):
    def __init__(self, file_path=None):
        BaseTsplibFileContainer.__init__(self, file_path)
        TxtFileRead.__init__(self, file_path)

    def process_file_data(self):
        rf_dict = self.create_reference_dict()
        temp = [[q.strip() for q in p] for p in [i.split(':')
                                                 for i in self.file_data]]
        flag = 0
        for line in temp:
            if len(line) == 2 and line[0] in rf_dict.keys():
                rf_dict[line[0]] = line[1]
            if flag == 1:
                rf_dict['NODE_COORD_SECTION'].append(line[0])
            if line[0] == 'NODE_COORD_SECTION':
                flag = 1
                rf_dict[line[0]] = []
        rf_dict['NODE_COORD_SECTION'] = rf_dict['NODE_COORD_SECTION'][:-1]
        if rf_dict['EDGE_WEIGHT_TYPE'] == 'EUC_2D':
            coords = {i[0]: [float(i[1]), float(i[2])] for i in [
                j.split() for j in rf_dict['NODE_COORD_SECTION']]}
            self.additional_data_container['coords'] = coords
            # print(self.additional_data_container['coords'])

    def create_reference_dict(self):
        D = {}
        keys = ['NAME', 'COMMENT', 'TYPE', 'DIMENSION',
                'EDGE_WEIGHT_TYPE', 'NODE_COORD_SECTION']
        for k in keys:
            D[k] = None
        return D


class TsplibOptDataContainer(BaseOptimalDataContainer, TxtFileRead):
    def __init__(self, file_path=None):
        BaseOptimalDataContainer.__init__(self, file_path)
        TxtFileRead.__init__(self, file_path)

    def process_file_data(self):
        # print(self.file_data)
        # Might have to change
        self.dataset_name = self.file_path.split('.')[0].split('/')[-1]
        ###
        self.optimal_route_data['cost'] = float(self.file_data[-1])
        opt_r = [i for i in self.file_data[6:-2]]
        self.optimal_route_data['route'] = opt_r


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    obj = TsplibTxtFileContainer('data/TSPLIB/Symmetric/TXT/a280.tsp')
