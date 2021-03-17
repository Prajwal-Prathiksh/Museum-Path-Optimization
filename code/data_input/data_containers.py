###########################################################################
# Imports
###########################################################################
# Standard library imports
import os
import numpy as np

from base_data_containers import BaseTsplibFileContainer, BaseOptimalDataContainer
from file_reader import XMLFileRead, NpzFileRead, TxtFileRead
###########################################################################
# Code
###########################################################################


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
                    vert_indx, (i-1)//2
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


class TsplibOptDataContainer(BaseOptimalDataContainer, TxtFileRead):
    def __init__(self, file_path=None):
        BaseOptimalDataContainer.__init__(self, file_path)
        TxtFileRead.__init__(self, file_path)

    def process_file_data(self):
        # Might have to change
        self.dataset_name = self.file_path.split('.')[0].split('/')[-1]
        ###
        self.optimal_route_data['cost'] = float(self.file_data[-1])
        opt_r = [i[:-1] for i in self.file_data[6:-2]]
        self.optimal_route_data['route'] = opt_r
