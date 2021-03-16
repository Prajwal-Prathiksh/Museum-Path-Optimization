import os
import sys
sys.path.insert(0, os.getcwd())


class Read_opt_cost:
    def __init__(self, path_to_opt_cost_file):
        self.read_data = self.read_fil(path_to_opt_cost_file)
        self.opt_cost_map = {}
        self.generate_opt_cost_map()

    def read_fil(self, path):
        f = open(path, 'r')
        data = f.readlines()
        f.close()
        return data

    def generate_opt_cost_map(self):
        for i in self.read_data:
            l = i.split(' : ')
            self.opt_cost_map[l[0]] = float(l[1].split(' ')[0][:-1])

    def get_opt_cost(self, dataset_name):
        if dataset_name in self.opt_cost_map.keys():
            return self.opt_cost_map[dataset_name]


if __name__ == '__main__':
    obj = Read_opt_cost('Sym_TSPLIB_opt_costs')
    print(obj.get_opt_cost('a280'))
