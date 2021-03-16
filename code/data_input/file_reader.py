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
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_data = []
        self.data_container = []
        if file_path != None:
            self.read_file()
            self.process_file_data()

    def read_file(self):
        pass

    def process_file_data(self):
        pass


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
        f.close()


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    ob = TxtFileRead('data_input/params/AllTC_XML')
