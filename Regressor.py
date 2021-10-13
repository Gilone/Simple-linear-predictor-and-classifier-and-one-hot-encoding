from DataMiner import DataMiner
import numpy as np

class RegressionTask(DataMiner):
    def __init__(self):
        self._book_review_data_dict_list= self._get_book_review_data_dict_list()
        self._weekday_one_hot_embedding, self._year_one_hot_embedding = self._get_week_year_one_hot_embedding()

    def _get_book_review_data_dict_list(self):
        data_dict_list = []
        for line_dict in self._parse_data('filename.json'):   # the data is like (date, review text) --> rating
            data_dict_list.append(line_dict)
        return data_dict_list

    def _get_week_year_one_hot_embedding(self):
        all_weekday_list = []
        all_year_list = []
        for line_dict in self._book_review_data_dict_list:
            weekday, year = self.get_date(line_dict['date'])
            all_weekday_list.append(weekday)
            all_year_list.append(year)
        return self.get_one_hot_embedding(all_weekday_list), self.get_one_hot_embedding(all_year_list)

    def _review_length_predictor(self, data_dict_list):
        X = []
        y = []
        for line_dict in data_dict_list:
            X.append([1, len(line_dict['review'])])
            y.append(line_dict['rating'])
        theta,residuals,rank,s = np.linalg.lstsq(X, y, rcond=None)
        mse = self.get_mse(X, y, theta)
        return theta, mse

    def _get_direct_length_date_feature(self, X, y, line_dict):
        weekday, year = self.get_date(line_dict['date'])
        X.append([1, len(line_dict['review']), weekday, year])
        y.append(line_dict['rating'])

    def _get_length_date_feature(self, X, y, line_dict):
        weekday, year = self.get_date(line_dict['date'])
        date_code_list = self.get_one_hot_code_from_embedding(weekday, self._weekday_one_hot_embedding)\
            + self.get_one_hot_code_from_embedding(year, self._year_one_hot_embedding)
        X.append([1, len(line_dict['review'])] + date_code_list)
        y.append(line_dict['rating'])

    def _directly_review_length_date_predictor(self, data_dict_list, splited_proportion=0):
        X = []
        y = []
        X_test = []
        y_test = []
        if splited_proportion == 0:
            for line_dict in data_dict_list:
                self._get_direct_length_date_feature(X, y, line_dict)
            theta,residuals,rank,s = np.linalg.lstsq(X, y, rcond=None)
            mse = [self.get_mse(X, y, theta)]
        else:
            train_data, test_data = self.split_data(data_dict_list, splited_proportion)
            for line_dict in train_data:
                self._get_direct_length_date_feature(X, y, line_dict)    
            for line_dict in test_data:
                self._get_direct_length_date_feature(X_test, y_test, line_dict)
            theta,residuals,rank,s = np.linalg.lstsq(X, y, rcond=None)
            mse_training = self.get_mse(X, y, theta)
            mse_test = self.get_mse(X_test, y_test, theta)
            mse = [mse_training, mse_test]
        return theta, mse

    def _review_length_date_predictor(self, data_dict_list, splited_proportion=0):
        X = []
        y = []
        X_test = []
        y_test = []
        if splited_proportion == 0:
            for line_dict in data_dict_list:
                self._get_length_date_feature(X, y, line_dict)
            theta,residuals,rank,s = np.linalg.lstsq(X, y, rcond=None)
            mse = [self.get_mse(X, y, theta)]
        else:
            train_data, test_data = self.split_data(data_dict_list, splited_proportion)
            for line_dict in train_data:
                self._get_length_date_feature(X, y, line_dict)    
            for line_dict in test_data:
                self._get_length_date_feature(X_test, y_test, line_dict)
            theta,residuals,rank,s = np.linalg.lstsq(X, y, rcond=None)
            mse_training = self.get_mse(X, y, theta)
            mse_test = self.get_mse(X_test, y_test, theta)
            mse = [mse_training, mse_test]
        return theta, mse

    def run(self):
        print("Train and test splited data direct MSE: ", \
            self._directly_review_length_date_predictor(self._book_review_data_dict_list, splited_proportion=0.5)[1])

        print("Train and test splited data one-hot MSE: ", \
            self._review_length_date_predictor(self._book_review_data_dict_list, splited_proportion=0.5)[1])