import numpy as np
import random
import dateutil.parser

class DataMiner():
    def __init__(self):
        pass

    def _parse_data(self, file_path):
        for line in open(file_path):
            yield eval(line)

    @staticmethod
    def get_mse(X, y, theta):
        X = np.array(X)
        return ((np.array(y) - np.dot(np.array(X), np.array(theta)))**2).mean()

    @staticmethod
    def get_date(data):
        t = dateutil.parser.parse(data)
        return int(t.weekday()), int(t.year)

    @staticmethod
    def get_one_hot_embedding(all_data_list):
        code = {}
        unique_list = sorted(list(set(all_data_list)))
        for i, key in enumerate(unique_list):
            code[key]=i
        return [len(unique_list)-1, code]

    @staticmethod
    def get_one_hot_code_from_embedding(data, embedding):
        code = []
        idx = embedding[1][data]
        for i in range(embedding[0]):
            code.append(0)
        if idx == 0:
            return code
        else:
            code[idx-1] = 1
        return code

    @staticmethod
    def split_data(data_list, proportion):
        random.shuffle(data_list)
        split_len = int(proportion*len(data_list))
        train_data = data_list[:split_len]
        test_data = data_list[split_len:]
        return train_data, test_data