###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
from xml.dom import minidom
import numpy as np

###########################################################################
# Code
###########################################################################


class FileReadContainer:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_data = []
        self.data_container = []
        if file_path is not None:
            self.read_file()
            self.process_file_data()

    def read_file(self):
        pass

    def process_file_data(self):
        pass

    def get_file_data(self):
        return self.file_data


class XMLFileRead(FileReadContainer):
    def __init__(self, file_path=None):
        FileReadContainer.__init__(self, file_path)

    def read_file(self):
        self.file_data = minidom.parse(self.file_path)


class NpzFileRead(FileReadContainer):
    def __init__(self, file_path=None):
        FileReadContainer.__init__(self, file_path)

    def read_file(self):
        self.file_data = np.load(self.file_path)


class TxtFileRead(FileReadContainer):
    def __init__(self, file_path=None):
        FileReadContainer.__init__(self, file_path)

    def read_file(self):
        f = open(self.file_path, 'r')
        self.file_data = f.readlines()
        for i in range(len(self.file_data)):
            if self.file_data[i][-1] == '\n':
                self.file_data[i] = self.file_data[i][:-1]
        f.close()
        # print(self.file_data)


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    ob = TxtFileRead('data_input/params/AllTC_XML')
