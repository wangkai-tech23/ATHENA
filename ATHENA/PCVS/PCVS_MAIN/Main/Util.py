import sys

import pandas as pd
from sklearn.metrics import confusion_matrix
from RuleMining import MISTree
from RuleMining import RuleMining
import numpy as np

def evaluate_prediction(actual_result,predict_result, verbose = 1):
    cmatrix = confusion_matrix(actual_result, predict_result)
    cmatrix = np.array([[cmatrix[1, 1], cmatrix[1, 0]], [cmatrix[0, 1], cmatrix[0, 0]]])
    tp, fn, fp, tn = cmatrix.ravel()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    if verbose == 1:
        print('precision: ' + str(precision))
        print('recall: ' + str(recall))
        print('f1score: ' + str(f1score))
        print('accuracy: ' + str(accuracy))

    return precision, recall, f1score, accuracy


def conInvarEntry(target_var, threshold, lessOrGreater, max_dict, min_dict, coefs, active_vars):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    msg = ''
    msg += target_var + lessOrGreater 
    count = 0
    for i in range(len(coefs)):
        if(coefs[i] > 0):
            msg += ' ' + str(coefs[i]) + '*' + active_vars[i]
            count += 1
    if count > 0:
        msg += '+' 
    msg += str(threshold_value)
    return msg

def conMarginEntry(target_var, threshold, margin, max_dict, min_dict):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]  
    
    msg = ''
    if margin == 0:
        msg +=  target_var + '<' + str(threshold_value)
    else:
        msg +=  target_var + '>' + str(threshold_value)
    
    return msg   

def conRangeEntry(target_var, lb, ub, max_dict, min_dict):
    threshold_lb = lb*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    threshold_up = ub*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    
    msg = ''
    msg += str(threshold_lb) + '<' + target_var + '<' + str(threshold_up)
    
    return msg



def getRules(training_data, dead_entries, keyArray, mode, gamma, max_k, theta):
    data = training_data.copy()
#     print(len(data))
    for entry in dead_entries:
        data = data.drop(entry, 1)

    def contains_any(column_name, patterns):
        return any(pattern in column_name for pattern in patterns)

    patterns = ["2AB", "618", "041", "1AA"]

    columns_to_drop = [col for col in data.columns if contains_any(col, patterns)]
    data.drop(columns=columns_to_drop, inplace=True)

    for entry in data:
        if mode == 0:
            if 'cluster' in entry:
                data = data.drop(entry, 1)
        elif mode == 1:
            if 'cluster' not in entry:
                data = data.drop(entry, 1)
                data = data.astype('int64')

    index_dict = {}
    item_dict = {}
    minSup_dict = {}
    index = 100
    for entry in data:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    min_num = len(data)*theta

    printed_dtype = False
    for entry in data:
        minSup_dict[ index_dict[entry]  ] = max( gamma*len(data[data[entry] == 1]), min_num )
        if mode == 1 and not printed_dtype:
            print(data[entry].dtype)
            printed_dtype = True
        data.loc[data[entry]==1, entry] = index_dict[entry]
        print(f"After assignment in {entry}:", data[entry].unique())

    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)

    item_count_dict = MISTree.count_items(dataset)

    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)
    L = RuleMining.filterClosedPatterns(freq_patterns, support_data, item_count_dict, max_k, MIN)
    rules = RuleMining.generateRules(L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=1)

    valid_rules = []
    for rule in rules:
        valid = True
        for i in range(len(keyArray)):
            for key in keyArray[i]:
                belongAnteq = False
                belongConseq = False
                for item in rule[0]:
                    if key in item_dict[item]:
                        belongAnteq = True
                        break
                for item in rule[1]:
                    if key in item_dict[item]:
                        belongConseq = True
                        break
                if belongAnteq == True and belongConseq == True:
                    valid = False
                    break
        if valid == True:
            valid_rules.append(rule)
    
    return valid_rules, item_dict