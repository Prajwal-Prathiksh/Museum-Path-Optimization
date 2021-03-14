import numpy as np

from .file_reader import XMLFileRead


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
                self.cost_matrix[vert_indx, (i + ((i//2) >= (vert_indx))) // 2] = float(
                    vertices[vert_indx].childNodes[i]._get_attributes().items()[0][1])

        self.update_cost_data()

    def update_cost_data(self):
        self.cost_data_container['cost_matrix'] = self.cost_matrix


""" obj = TsplibXMLFileContainer('a280.xml')
for i in range(len(obj.cost_matrix)):
    print(obj.cost_matrix[i,i]) """
