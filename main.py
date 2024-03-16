import math
import random
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.metrics import auc

def define_balance(df: pd.DataFrame) -> str:
    return df['GT'].value_counts().to_string()


def calculate_metrics_data_frame(df: pd.DataFrame, thresholds_step: float) -> None:
    list_of_thresholds = [round(i * thresholds_step, 4) for i in range(int(1 / thresholds_step) + 1)]

    for model in [('Model_1_0', 'Model_1_1'), ('Model_2_0', 'Model_2_1')]:
        all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f_scores': [], 
                       'MCC': [], 'BA': [], 'Y_J_statistics': [], 'FPR': [], 'class_0': [], 'class_1': []}
        
        print("\n", model[:7])
        for threshold in list_of_thresholds:
            calculate_all_metrics(all_metrics, df, model, threshold)

        # 3.a
        print('Thresholds: ', list_of_thresholds)
        for key, value in all_metrics.items():
            print(f'{key}: {value}')
            
        # 3.b
        show_graph_all_metrics_with_thresholds(all_metrics, model, list_of_thresholds)
            
        # 3.c
        graphs_number_of_objects(all_metrics, model, list_of_thresholds)
        graphs_number_of_objects_and_mertics(all_metrics, model, list_of_thresholds)
            
        # 3.d
        show_graph_PRC(all_metrics['recall'], all_metrics['precision'], model)
        show_graph_ROC(all_metrics["FPR"], all_metrics['recall'], model)


def calculate_all_metrics(metrics: dict, df: pd.DataFrame, model: tuple, threshold: float)-> None:
    TP = len(df.loc[(df["GT"] == 1) & (df[model[1]] > threshold)])
    FP = len(df.loc[(df["GT"] == 0) & (df[model[1]] > threshold)])
    FN = len(df.loc[(df["GT"] == 1) & (df[model[0]] >= 1 - threshold)])
    TN = len(df.loc[(df["GT"] == 0) & (df[model[0]] >= 1 - threshold)])

    if(TP + TN + FP + FN == 0): return

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    try:
        precision = TP / (TP + FP) 
    except ZeroDivisionError:
        precision = 0

    try:
        recall = TP / (TP + FN) 
    except ZeroDivisionError:
        recall = 0

    try:
        FPR = FP / (FP + TN)
    except ZeroDivisionError:
        FPR = 0

    b = 1
    try:
        f_scores = (1 + b**2) * precision * recall / (b**2 * (precision + recall))
    except: 
        f_scores = 0

    try:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except ZeroDivisionError:
        MCC = 0

    try:
        BA = (TP / (TP + FN) + TN / (TN + FP)) / 2
    except ZeroDivisionError:
        BA = 0

    try:
        Y_J_statistics = TP / (TP + FN) + TN / (TN + FP) - 1
    except ZeroDivisionError:
        Y_J_statistics = 0

    metrics['accuracy'] += [round(accuracy, 4)]
    metrics['precision'] += [round(precision, 4)]
    metrics['recall'] += [round(recall, 4)]
    metrics['f_scores'] += [round(f_scores, 4)]
    metrics['MCC'] += [round(MCC, 4)]
    metrics['BA'] += [round(BA, 4)]
    metrics['Y_J_statistics'] += [round(Y_J_statistics, 4)]
    metrics['FPR'] += [round(FPR, 4)]
    metrics['class_0'].append(TN + FN)
    metrics['class_1'].append(TP + FP)


def graphs_number_of_objects_and_mertics(metrics: dict, model_name: tuple, thresholds: list[float]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    column_of_metrics = ['accuracy', 'precision', 'recall', 'f_scores', 'MCC', 'BA', 'Y_J_statistics']
    class_1_value = metrics['class_1']
    for i in column_of_metrics:
        metrics_value = metrics[i]

        metrics_max_value = max(metrics_value)
        max_value_index = metrics_value.index(metrics_max_value)

        axs[0].plot(class_1_value, metrics_value, label=i)
        axs[0].axvline(x=class_1_value[max_value_index], color='black', linestyle='--')
        axs[0].set_xlabel('Number of object: "Class 1"')
        axs[0].set_ylabel('Metric values')
        axs[0].legend()

    class_0_value = metrics['class_0']
    for i in column_of_metrics:
        metrics_value = metrics[i]

        metrics_max_value = max(metrics_value)
        max_value_index = metrics_value.index(metrics_max_value)

        axs[1].plot(class_0_value, metrics_value, label=i)
        axs[1].axvline(x=class_0_value[max_value_index], color='black', linestyle='--')
        axs[1].set_xlabel('Number of object: "Class 0"')
        axs[1].set_ylabel('Metric values')
        axs[1].legend()

    plt.title(f'{model_name[0][:7]}')
    plt.legend()
    plt.show()


def graphs_number_of_objects(metrics: dict, model_name: tuple, thresholds: list[float]) -> None:
    diff = [abs(d1 - d2) for d1, d2 in zip(metrics['class_1'], metrics['class_0'])]
    min_diff_index = diff.index(min(diff))

    plt.figure(figsize=(10, 6))
    plt.bar(thresholds, metrics['class_1'], width=0.04, color='blue', alpha=0.5, label='Сlass_1')
    plt.bar(thresholds, metrics['class_0'], width=0.04, color='red', alpha=0.5, label='Сlass_0')

    plt.axvline(x=thresholds[min_diff_index], color='green', linestyle='--', label='Min Difference')

    plt.xlabel('Threshold')
    plt.ylabel('Number of objects')
    plt.title(f'{model_name[0][:7]}')
    plt.legend()
    plt.show()


def show_graph_all_metrics_with_thresholds(metrics: dict, model_name: tuple, thresholds: list[float]) -> None:
    metrics_without_FPR = deepcopy(metrics)
    metrics_without_FPR.pop('FPR')
    metrics_without_FPR.pop('class_0')
    metrics_without_FPR.pop('class_1')

    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics_without_FPR.items():
        plt.plot(thresholds, values, label=metric_name)

        max_value = max(values)
        max_thresholds = thresholds[values.index(max_value)]
        plt.scatter(max_thresholds, max_value, marker='X', color='red')
        plt.text(max_thresholds, max_value, f"({max_thresholds}, {round(max_value, 3)})", fontsize=10)

    plt.xlim([-0.01, 1.25])
    plt.ylim([-0.01, 1.07])

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics for {model_name[0][:7]}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def show_graph_PRC(recall: list[float], precision: list[float], model_name: tuple) -> None:
    AUC_PRC = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, marker="o", color='blue', lw=2.5,
             label=f'PRC curve (area = {AUC_PRC:.3f})')

    plt.plot([0, 1], [0, 1], color='orange', lw=2.5, linestyle='-')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name[0][:7]})')
    plt.legend(loc='lower right')
    plt.show()


def show_graph_ROC(FPR: list[float], TPR: list[float], model_name: tuple) -> None:
    AUC_PRC = auc(FPR, TPR)

    plt.figure()
    plt.plot(FPR, TPR, marker="o", color='blue', lw=2.5,
             label=f'ROC curve (area = {AUC_PRC:.3f}')

    plt.plot([1, 0], [0, 1], color='orange', lw=2.5, linestyle='-')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({model_name[0][:7]})')
    plt.legend(loc='upper right')
    plt.show()


def new_data_frame(df: pd.DataFrame, birth_date: str):
    new_df = df.copy()
    birth_day = int(birth_date.split("-")[0])
    k = birth_day % 9
    percentage_values_to_delete = 50 + 5 * k
    print("Percentage of deleted objects:", percentage_values_to_delete)

    class_1 = new_df[new_df['GT'] == 1]
    num_rows_to_delete = int(len(class_1) * (percentage_values_to_delete / 100))
    rows_to_delete = random.sample(list(class_1.index), num_rows_to_delete)

    return new_df.drop(rows_to_delete)


if __name__ == '__main__':
    # 1
    df = pd.read_csv('KM-12-2.csv')

    # 2
    print(define_balance(df))

    # 3
    thresholds_step = 0.05

    calculate_metrics_data_frame(df, thresholds_step)

    # # 5
    # birth_date = '30-07'

    # new_df = new_data_frame(df, birth_date)

    # # 6
    # print(define_balance(new_df))

    # # 7
    # calculate_metrics_data_frame(new_df, thresholds_step)

