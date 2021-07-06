import csv
import math
import time
import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

gini_split_list = list()
gain_ratio_list = list()
info_gain_list = list()
error_list = list()
chi_square_list = list()


def feature_calculate(i):
    gini_split_list.clear()
    gain_ratio_list.clear()
    info_gain_list.clear()
    error_list.clear()
    chi_square_list.clear()
    fname = 'D:/User/Janhvi/BITS_4sem/BITS Y2S2/CS F415/Assignment/' + str(i) + '.csv'
    tdata = np.genfromtxt(fname, delimiter=',')

    mean = np.mean(tdata, axis=0)
    for col in range(len(tdata[0, 0:-1])):
        np.place(tdata[:, col], tdata[:, col] < mean[col], 0)
        np.place(tdata[:, col], tdata[:, col] >= mean[col], 1)
    np.place(tdata[:, -1], tdata[:, -1] > 0, 1)

    for col in range(len(tdata[0, 0:-1])):
        gini_split = 0
        split_info = 0
        info_gain = 0
        entropy_of_class = 0
        entropy_of_parent = 0
        error = 0
        chi_square = 0

        for i in np.unique(tdata[:, col]):
            uni_col = tdata[tdata[:, col] == i]
            total = len(uni_col)
            gini_index = 1

            for j in np.unique(tdata[:, -1]):
                uni_col_class = uni_col[uni_col[:, -1] == j]
                uni_class = tdata[tdata[:, -1] == j]

                prob = (len(uni_col_class) / total)

                observed_value = len(uni_col_class)
                expected_value = ((len(uni_col) * len(uni_class)) / len(tdata[:, 0]))
                chi_square += (pow((observed_value - expected_value), 2) / expected_value)

                error = max(error, prob)
                gini_index -= pow(prob, 2)
                if prob != 0:
                    entropy_of_class -= prob * math.log(prob)
                else:
                    entropy_of_class -= 0

            ni_by_n = len(uni_col) / len(tdata[:, 0])
            split_info -= ni_by_n * math.log(ni_by_n)
            gini_split += gini_index * ni_by_n
            info_gain -= (total / len(tdata[:, col])) * math.log(
                total / len(tdata[:, col])) - ni_by_n * entropy_of_class

        gain_ratio = np.divide(info_gain, split_info)
        error = 1 - error

        chi_square_list.append((chi_square, 'Feature ' + str(col)))
        error_list.append((error, 'Feature ' + str(col)))
        gain_ratio_list.append((gain_ratio, 'Feature ' + str(col)))
        gini_split_list.append((gini_split, 'Feature ' + str(col)))
        info_gain_list.append((info_gain, 'Feature ' + str(col)))

    print(np.shape(tdata))


def make_tree(meth, data, originaldata, features, target_attribute_name="class_type", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class


    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

    item_values = list()
    if meth == 'ID3':
        for feature in features:
            item_values.append(info_gain_list[int(feature[8:])][0])
        best_feature_index = np.argmax(item_values)

    elif meth == 'chaid':
        for feature in features:
            item_values.append(chi_square_list[int(feature[8:])][0])
        best_feature_index = np.argmax(item_values)

    elif meth == 'misclassificationerror':
        for feature in features:
            item_values.append(error_list[int(feature[8:])][0])
        best_feature_index = np.argmin(item_values)

    elif meth == 'c4point5':
        for feature in features:
            item_values.append(gain_ratio_list[int(feature[8:])][0])
        best_feature_index = np.argmax(item_values)

    elif meth == 'cart':
        for feature in features:
            item_values.append(gini_split_list[int(feature[8:])][0])
        best_feature_index = np.argmin(item_values)

    best_feature = features[best_feature_index]

    tree = {best_feature: {}}

    features = [i for i in features if i != best_feature]


    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = make_tree(meth, sub_data, dataset, features, target_attribute_name, parent_node_class)
        tree[best_feature][value] = subtree
    return (tree)


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def train_test_split(dataset, k, i):
    n = len(dataset)
    training_data1 = dataset.iloc[:int((i * n) / k)].reset_index(drop=True)
    testing_data = dataset.iloc[int((i * n) / k):int(((i + 1) * n) / k)].reset_index(drop=True)
    training_data2 = dataset.iloc[int(((i + 1) * n) / k):].reset_index(drop=True)
    training_data = training_data1.append(training_data2)

    return training_data, testing_data


def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1)

    a, b, c, d = 0, 0, 0, 0
    for i in range(len(data)):
        if predicted["predicted"][i] == 1:
            if data["class_type"][i] == 1:
                a += 1
            else:
                c += 1
        else:
            if data["class_type"][i] == 1:
                b += 1
            else:
                d += 1

    accuracy = (a + d) / (a + b + c + d)
    if 2 * a + b + c != 0:
        F_Measure = 2 * a / (2 * a + b + c)
    else:
        F_Measure = math.inf

    return accuracy, F_Measure


methods = ['chaid', 'misclassificationerror', 'c4point5', 'cart', 'ID3']
st = time.time()

for data_file in range(1, 4):

    print(data_file, end=' ')

    fname = 'D:/User/Janhvi/BITS_4sem/BITS Y2S2/CS F415/Assignment/' + str(data_file) + '.csv'
    tempdata = np.genfromtxt(fname, delimiter=',')
    columns = list()
    for i in range(20):
        columns.append('Feature ' + str(i))
    columns.append('class_type')
    feature_calculate(data_file)
    dataset = pd.DataFrame(data=tempdata, columns=columns)

    accu = list()
    f_mea = list()

    k = 10

    for meth in methods:
        csv_fname = str(meth) + '.csv'
        for i in range(k):
            training_data = train_test_split(dataset, 10, i)[0]
            testing_data = train_test_split(dataset, 10, i)[1]

            tree = make_tree(meth, training_data, training_data, training_data.columns[:-1])

            a, f = test(testing_data, tree)
            accu.append(a)
            f_mea.append(f)

        values = list()
        values.append(float(sum(accu) / len(accu)))
        values.append(float(sum(f_mea) / len(f_mea)))

        csv_file = open(csv_fname, 'a')
        writer = csv.writer(csv_file, lineterminator='\r')
        writer.writerow(values)
        csv_file.close()

    print('Done')

print(time.time() - st)
