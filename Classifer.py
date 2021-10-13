from DataMiner import DataMiner
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

class ClassificationTask(DataMiner):
    def __init__(self):
        self._beer_review_data_dict_list= self._get_beer_review_data_dict_list()

    def _get_beer_review_data_dict_list(self):
        data_dict_list = []
        for line_dict in self._parse_data('finename.json'): # the data is like (review text) --> rating
            data_dict_list.append(line_dict)
        return data_dict_list

    def _get_result_rates(self, pred, ytest):
        TP = sum(np.logical_and(pred, ytest))
        TN = sum(np.logical_and(np.logical_not(pred), np.logical_not(ytest)))
        FP = sum(np.logical_and(pred, np.logical_not(ytest)))
        FN = sum(np.logical_and(np.logical_not(pred), ytest))
        BER = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))
        return TP, TN, FP, FN, BER

    def _get_top_k_precision_list(self, scores, y_test, range_num):
        scores_labels = list(zip(scores, y_test))
        scores_labels.sort(reverse = True)
        top_k_precision_list = []
        y_test_list = []
        for k in range(range_num):
            y_test_list.append(scores_labels[k][1])
            top_k_precision_list.append(sum(y_test_list)/(k+1))
        return top_k_precision_list

    def _get_binary_top_k_precision_list(self, scores, y_test, pred, range_num):
        binary_scores = [abs(s) for s in scores]
        correct_values = [1 if y_test[k] == pred[k] else 0 for k in range(len(scores))]
        scores_labels = list(zip(binary_scores, correct_values))
        scores_labels.sort(reverse=True)
        sorted_correct_values = [x[1] for x in scores_labels]
        binary_top_k_precision_list = [sum(sorted_correct_values[:k+1])/(k+1) for k in range(range_num)]
        return binary_top_k_precision_list

    def _review_length_logistic_regressor(self):
        mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
        train_data = self._beer_review_data_dict_list
        X = []
        y = []
        for line_dict in train_data:
            X.append([1, len(line_dict['review'])])
            y.append(line_dict['rating']>= 4)
        mod.fit(X, y)
        test_predictions = mod.predict(X)
        scores = mod.decision_function(X)
        self._top_k_precision_list = self._get_top_k_precision_list(scores, y, 10000)
        self._binary_top_k_precision_list = self._get_binary_top_k_precision_list(scores, y, test_predictions, 10000)
        return self._get_result_rates(test_predictions, y)

    def _plot_pk(self):
        k_list = []
        top_k_list = []
        for k in range(0, 10000, 10):
            k_list.append(k+1)
            top_k_list.append(self._top_k_precision_list[k])
        plt.plot(k_list, top_k_list, ls='-', lw=2, label='precision', color='purple')
        plt.legend()
        plt.xlabel('K')
        plt.ylabel('precision@K')
        print("\n---Question 8---")
        plt.show()

    def _plot_bpk(self):
        k_list = []
        top_k_list = []
        for k in range(0, 10000, 10):
            k_list.append(k+1)
            top_k_list.append(self._binary_top_k_precision_list[k])
        plt.plot(k_list, top_k_list, ls='-', lw=2, label='precision', color='red')
        plt.legend()
        plt.xlabel('K')
        plt.ylabel('binary_precision@K')
        print("\n---Top K at 1, 100, 10000---")
        print("K=1: ", self._binary_top_k_precision_list[0], \
            ", K=100: ", self._binary_top_k_precision_list[99], \
            ", K=10000: ", self._binary_top_k_precision_list[9999])
        plt.show()

    def run(self):
        print("TP, TN, FP, FN, BER: ", self._review_length_logistic_regressor())
        self._plot_pk()
        self._plot_bpk()
