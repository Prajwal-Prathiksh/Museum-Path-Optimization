from xml.dom import minidom

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


""" obj = XMLFileRead('a280.xml')
print(obj.file_data) """