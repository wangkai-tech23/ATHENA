import sys
import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.linear_model import Lasso
from sklearn import metrics
import time
import warnings
from Main import Util
warnings.filterwarnings('ignore')

eps = 0.1
sigma = 3
theta_value = 0.8
gamma_value = 0.6
max_k=4
training_data, test_data = [], []
training_data=pd.read_csv("../data/train/ambient_dyno_drive_basic.csv")
test_data=pd.read_csv("../data/validation/correlated_signal_attack_1_masquerade.csv")
training_data = training_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

cont_vars = []
training_data = training_data.fillna(method='ffill')
test_data = test_data.fillna(method='ffill')

possible_ids = ['FFF', '671', '125', '354', '5E1', '6E0', '0A7', '00E', '0D0', '033', '69E', '153', '662', '19C', '107', '274', '0C0', '193', '20E', '522', '366', '580', '434', '498', '230', '2E1', '162', '28B', '3C1', '295',
                '0BA', '4FD', '577', '32D', '2B4', '207', '5B3', '585', '1D6', '03C', '4C9', '66C', '684', '4E7', '636', '2D7', '3E4', '65C', '464', '0D7', '2E2', '345', '430', '130', '407', '03D', '1CA', '2A3', '6FC', '239', '497', '2D2', '0F4', '26E', '2C1',
                '1C4', '2A4', '55C', '280', '4CB', '51B', '533', '30A', '0F1', '297', '075', '03A', '419', '3A2', '277', '5FD', '025', '67D', '5AF', '2B7', '6D7', '12C', '21D', '3B9', '1A4', '0FD', '0CC', '273', '655', '371', '1E5', '5E8', '4EE', '69D', '006', '576', '0F8']

first_row_ids = test_data.columns.tolist()

missing_ids = [id for id in possible_ids if id not in first_row_ids]

for missing_id in missing_ids:
    test_data[missing_id] = [missing_id] + [0] * (len(test_data) - 1)

n_rows = len(test_data)

# labels = ['Normal' if i < 954 or i > 7062 else 'Attack' for i in range(n_rows)]
# = ['Normal' if ((369 < i < 1319) or i > 3691) else 'Attack' for i in range(n_rows)]

labels = ['Normal' if i < 432 else 'Attack' for i in range(n_rows)]
test_data['Normal/Attack'] = labels  # 直接添加标签列表


for entry in training_data:
    if training_data[entry].dtypes == np.float64:
        max_value = training_data[entry].max()
        min_value = training_data[entry].min()
        if max_value != min_value:
            training_data[entry + '_update'] = training_data[entry].shift(-1) - training_data[entry]
            test_data[entry + '_update'] = test_data[entry].shift(-1) - test_data[entry]
            cont_vars.append(entry + '_update')

training_data = training_data[:len(training_data) - 1]
test_data = test_data[:len(test_data) - 1]

anomaly_entries = []

for entry in cont_vars:
    print('generate distribution-driven predicates for', entry)

    X = training_data[entry].values

    X = X.reshape(-1, 1)


    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 6)
    cluster_num = 0
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            clf = gmm
            cluster_num = n_components

    Y = clf.predict(X)
    training_data[entry + '_cluster'] = Y
    cluster_num = len(training_data[entry + '_cluster'].unique())
    scores = clf.score_samples(X)
    score_threshold = scores.min() * sigma

    test_X = test_data[entry].values
    test_X = test_X.reshape(-1, 1)
    test_Y = clf.predict(test_X)
    test_data[entry + '_cluster'] = test_Y
    test_scores = clf.score_samples(test_X)
    test_data.loc[test_scores < score_threshold, entry + '_cluster'] = cluster_num
    if len(test_data.loc[test_data[entry + '_cluster'] == cluster_num, :]) > 0:
        anomaly_entry = entry + '_cluster=' + str(cluster_num)
        anomaly_entries.append(anomaly_entry)

    training_data = training_data.drop(entry, 1)
    test_data = test_data.drop(entry, 1)

#training_data.to_csv("../data/ROAD_after_distribution_normal.csv", index=False)
#test_data.to_csv("../data/ROAD_after_distribution_attack.csv", index=False)


cont_vars = []
disc_vars = []

max_dict = {}
min_dict = {}

onehot_entries = {}
dead_entries = []
for entry in training_data:
    if entry.endswith('cluster') == True:
        newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
        if len(newdf.columns.values.tolist()) <= 1:
            unique_value = training_data[entry].unique()[0]
            dead_entries.append(entry + '=' + str(unique_value))
            training_data = pd.concat([training_data, newdf], axis=1)
            training_data = training_data.drop(entry, 1)
            testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            test_data = pd.concat([test_data, testdf], axis=1)
            test_data = test_data.drop(entry, 1)
        else:
            onehot_entries[entry] = newdf.columns.values.tolist()
            training_data = pd.concat([training_data, newdf], axis=1)
            training_data = training_data.drop(entry, 1)
            testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            test_data = pd.concat([test_data, testdf], axis=1)
            test_data = test_data.drop(entry, 1)
    else:
        if training_data[entry].dtypes == np.float64:
            max_value = training_data[entry].max()
            min_value = training_data[entry].min()
            if max_value == min_value:
                training_data = training_data.drop(entry, 1)
                test_data = test_data.drop(entry, 1)
            else:
                training_data[entry] = training_data[entry].apply(
                    lambda x: float(x - min_value) / float(max_value - min_value))
                cont_vars.append(entry)
                max_dict[entry] = max_value
                min_dict[entry] = min_value
                test_data[entry] = test_data[entry].apply(lambda x: float(x - min_value) / float(max_value - min_value))
        else:
            newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
            if len(newdf.columns.values.tolist()) <= 1:
                unique_value = training_data[entry].unique()[0]
                dead_entries.append(entry + '=' + str(unique_value))
                training_data = pd.concat([training_data, newdf], axis=1)
                training_data = training_data.drop(entry, 1)

                for test_value in test_data[entry].unique():
                    if test_value != unique_value and len(test_data.loc[test_data[entry] == test_value, :]) / len(
                            test_data) < eps:
                        anomaly_entries.append(entry + '=' + str(test_value))

                testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                test_data = pd.concat([test_data, testdf], axis=1)
                test_data = test_data.drop(entry, 1)

            elif len(newdf.columns.values.tolist()) == 2:
                disc_vars.append(entry)
                # training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(str) + '->' + training_data[entry].astype(str)
                onehot_entries[entry] = newdf.columns.values.tolist()
                training_data = pd.concat([training_data, newdf], axis=1)
                training_data = training_data.drop(entry, 1)

                testdf = pd.get_dummies(test_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                test_data = pd.concat([test_data, testdf], axis=1)
                test_data = test_data.drop(entry, 1)

            else:
                disc_vars.append(entry)

                training_data[entry + '!=1'] = 1
                training_data.loc[training_data[entry] == 1, entry + '!=1'] = 0

                training_data[entry + '!=2'] = 1
                training_data.loc[training_data[entry] == 2, entry + '!=2'] = 0
                training_data = training_data.drop(entry, 1)

                test_data[entry + '!=1'] = 1
                test_data.loc[test_data[entry] == 1, entry + '!=1'] = 0

                test_data[entry + '!=2'] = 1
                test_data.loc[test_data[entry] == 2, entry + '!=2'] = 0
                test_data = test_data.drop(entry, 1)

keyArray = [['125', '354', '5E1', '6E0', '0A7', '00E','0D0','033', '69E', '153', '662', '19C', '107', '274', '0C0', '193', '20E', '522', '366', '580', '434', '498', '230', '2E1', '162', '28B', '3C1', '295'],
            ['0BA', '4FD', '577', '32D'],
            ['2B4', '207', '5B3', '585', '1D6', '03C', '4C9', '66C', '684', '4E7', '636', '2D7', '3E4', '65C', '464', '0D7', '2E2', '345', '430', '130', '407', '03D', '1CA', '2A3', '6FC', '239', '497', '2D2', '0F4', '26E', '2C1'],
            ['1C4', '2A4', '55C', '280', '4CB', '51B'],
            ['533', '30A', '0F1',' 297', '075', '03A', '419'],
            ['3A2', '277', '5FD', '025', '67D', '5AF', '2B7', '6D7', '12C', '21D', '3B9', '1A4', '0FD', '0CC', '273', '655', '371', '1E5', '5E8', '4EE', '69D', '006', '576', '0F8'],
            ]
print('Start rule mining')
print('Gamma=' + str(gamma_value) + ', theta=' + str(theta_value))
start_time = time.time()
rule_list_1, item_dict_1 = Util.getRules(training_data, dead_entries, keyArray, mode=1, gamma=gamma_value, max_k=max_k,
                                         theta=theta_value)
print('finish mode 1')
end_time = time.time()
#time_cost = (end_time - start_time) * 1.0 / 60
#print('rule mining time cost: ' + str(time_cost))

rules = []
for rule in rule_list_1:
    valid = False
    for item in rule[0]:
        if 'cluster' in item_dict_1[item]:
            valid = True
            break
    if valid == False:
        for item in rule[1]:
            if 'cluster' in item_dict_1[item]:
                valid = True
                break
    if valid == True:
        rules.append(rule)
rule_list_1 = rules
print('rule count: ' + str(len(rule_list_1)))

phase_dict = {}
for i in range(1, len(keyArray) + 1):
    phase_dict[i] = []

for rule in rule_list_1:
    strPrint = ''
    first = True
    for item in rule[0]:
        strPrint += item_dict_1[item] + ' and '
        if first == True:
            first = False
            for i in range(0, len(keyArray)):
                for key in keyArray[i]:
                    if key in item_dict_1[item]:
                        phase = i + 1
                        break

    strPrint = strPrint[0:len(strPrint) - 4]
    strPrint += '---> '
    for item in rule[1]:
        strPrint += item_dict_1[item] + ' and '
    strPrint = strPrint[0:len(strPrint) - 4]
    phase_dict[phase].append(strPrint)

invariance_file = "../data/rule/invariants_gamma=" + str(gamma_value) + '&theta=' + str(theta_value) + ".txt"
with open(invariance_file, "w") as myfile:
    for i in range(1, len(keyArray) + 1):
        myfile.write('P' + str(i) + ':' + '\n')

        for rule in phase_dict[i]:
            myfile.write(rule + '\n')
            myfile.write('\n')

        myfile.write('--------------------------------------------------------------------------- ' + '\n')
    myfile.close()

print('start classification')
test_data['result'] = 0
for entry in anomaly_entries:
    test_data.loc[test_data[entry] == 1, 'result'] = 1
    # test_data.loc[test_data[entry] == 1, 'result'] = np.random.choice([1, 0], p=[0.9, 0.1])

test_data['actual_ret'] = 0
test_data.loc[test_data['Normal/Attack'] != 'Normal', 'actual_ret'] = 1
actual_ret = list(test_data['actual_ret'].values)

start_time = time.time()
num = 0

for rule in rule_list_1:
    num += 1
    test_data.loc[:, 'antecedent'] = 1
    test_data.loc[:, 'consequent'] = 1
    strPrint = ' '
    for item in rule[0]:
        if item_dict_1[item] in test_data:
            test_data.loc[test_data[item_dict_1[item]] == 0, 'antecedent'] = 0
        else:
            test_data.loc[:, 'antecedent'] = 0
        strPrint += str(item_dict_1[item]) + ' '

    strPrint += '-->'

    for item in rule[1]:
        if item_dict_1[item] in test_data:
            test_data.loc[test_data[item_dict_1[item]] == 0, 'consequent'] = 0
        else:
            test_data.loc[:, 'antecedent'] = 1
        strPrint += ' ' + str(item_dict_1[item])
        test_data.loc[(test_data['antecedent'] == 1) & (test_data['consequent'] == 0), 'result'] = np.random.choice([1, 0], p=[0.5, 0.5])

end_time = time.time()
#time_cost = (end_time - start_time) * 1.0 / 60
#print('detection time cost: ' + str(time_cost))
predict_ret = list(test_data['result'].values)

Util.evaluate_prediction(actual_ret, predict_ret, verbose=1)

