import os
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import precision_score, recall_score, ndcg_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def normalize_label_score(label_scores, eps=1e-7):
    label_scores = -label_scores
    scores_clean = np.nan_to_num(label_scores, nan=0.0, posinf=None, neginf=None)
    # 异常值掩码
    mask_inf = (scores_clean < -1e100) | (scores_clean > 1e100) 
    # 全局计算min/max
    min_ = np.where(mask_inf, float('inf'), scores_clean).min()
    max_ = np.where(mask_inf, -float('inf'), scores_clean).max()
    range_minmax = max(max_ - min_, eps)
    normalized = (scores_clean - min_) / range_minmax
    normalized = np.clip(normalized, a_min=0.0, a_max=1.0)
    # 将矩阵中的np.inf替换为0（负样本）
    normalized[np.isinf(normalized)] = 0.0
    return normalized

def precision_at_k(true_labels, pred_scores, k, threshold=0.5):
    precision_scores = []
    for true_labels_single, pred_scores_single in zip(true_labels, pred_scores):
        k_actual = min(k, len(true_labels_single))  # Ensure k is not larger than the number of true labels
        sorted_indices = np.argsort(-pred_scores_single)  # Sort indices by predicted scores in descending order
        top_k_indices = sorted_indices[:k_actual]  # Get the top-k indices
        
        # Convert the top-k scores to binary predictions based on the threshold (0.5 in this case)
        top_k_predictions = np.zeros_like(true_labels_single)
        top_k_predictions[top_k_indices] = (pred_scores_single[top_k_indices] > threshold).astype(int)
        
        # Calculate precision: the fraction of correct predictions in the top-k items
        precision = precision_score(true_labels_single, top_k_predictions)
        precision_scores.append(precision)
    
    return np.mean(precision_scores)  # Return the average precision across all samples
    
def mean_reciprocal_rank(true_labels, pred_scores):
    mrr_scores = []
    for true_labels_single, pred_scores_single in zip(true_labels, pred_scores):
        # 排序推荐分数
        sorted_indices = np.argsort(-pred_scores_single)  # 降序排列
        for rank, idx in enumerate(sorted_indices, start=1):
            if true_labels_single[idx] == 1:  # 找到第一个相关物品
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0)  # 如果没有相关物品，则 MRR 为 0
    return np.mean(mrr_scores)

def recall_at_k(true_labels, pred_scores, k):
    recall_scores = []
    for true_labels_single, pred_scores_single in zip(true_labels, pred_scores):
        # 排序推荐分数
        sorted_indices = np.argsort(-pred_scores_single)  # 降序排列
        top_k_indices = sorted_indices[:k]
        
        # 计算 Recall@K
        recall = recall_score(true_labels_single, np.array([1 if i in top_k_indices else 0 for i in range(len(true_labels_single))]))
        recall_scores.append(recall)
    
    return np.mean(recall_scores)

def evaluate_performance(data): #pred, 
    data = data
    data['normalized_score'] = normalize_label_score(data['score'].values)
    # data['pred_score'] = pred
    grouped = data.groupby('gene1')
    true_scores_per = []
    pred_scores_per = []
    true_label_per = []
    for gene1, group in grouped:
        # group['normalized_score'] = normalize_label_score(group['score'].values)
        group_data = group[['label', 'normalized_score', 'pred_score']].values
        true_label_per.append(group_data[:,0])
        true_scores_per.append(group_data[:,1]) 
        pred_scores_per.append(group_data[:,2]) 
    true_scores_per = np.vstack(true_scores_per)
    pred_scores_per= np.vstack(pred_scores_per)
    true_label_per=np.vstack(true_label_per)

    ndcg_1 = ndcg_score(true_scores_per, pred_scores_per, k=1)
    ndcg_5 = ndcg_score(true_scores_per, pred_scores_per, k=5)
    ndcg_10 = ndcg_score(true_scores_per, pred_scores_per, k=10)
    ndcg_15 = ndcg_score(true_scores_per, pred_scores_per, k=15)
    ndcg_20 = ndcg_score(true_scores_per, pred_scores_per, k=20)

    mrr = mean_reciprocal_rank(true_label_per, pred_scores_per)

    # precision_1 = precision_at_k(true_label_per, pred_scores_per, k=1)
    # precision_5 = precision_at_k(true_label_per, pred_scores_per, k=5)
    # precision_10 = precision_at_k(true_label_per, pred_scores_per, k=10)

    recall_1 = recall_at_k(true_label_per, pred_scores_per, k=1)
    recall_5 = recall_at_k(true_label_per, pred_scores_per, k=5)
    recall_10 = recall_at_k(true_label_per, pred_scores_per, k=10)
    recall_15 = recall_at_k(true_label_per, pred_scores_per, k=15)
    recall_20 = recall_at_k(true_label_per, pred_scores_per, k=20)
    
    return {
        'ndcg_1': ndcg_1,
        'ndcg_5': ndcg_5,
        'ndcg_10': ndcg_10,
        'ndcg_15': ndcg_15,
        'ndcg_20': ndcg_20,
        'mrr': mrr,
        # 'precision_1': precision_1,
        # 'precision_5': precision_5,
        # 'precision_10': precision_10,
        'recall_1': recall_1,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'recall_15': recall_15,
        'recall_20': recall_20
    }

####################################################################
output_root = '/data/qingyuyang/ESM4SL/output/attn'

# cell_lines = ['30033366']
# scenes = ['Jurkat_scenario1', 'Jurkat_scenario2', 'K562_scenario1', 'K562_scenario2']
cell_lines = ['28319113']
scenes = ['293T_scenario1', '293T_scenario2', 'A549_scenario1', 'A549_scenario2', 'HELA_scenario1', 'HELA_scenario2']

column_names = [
    'ndcg_1', 'ndcg_5', 'ndcg_10', 'ndcg_15', 'ndcg_20',
    'mrr',
    # 'precision_1', 'precision_5', 'precision_10',
    'recall_1', 'recall_5', 'recall_10', 'recall_15', 'recall_20'
]

for cell_line, scene in list(itertools.product(cell_lines, scenes)):
    print(f'======================== {cell_line} / {scene} ========================')

    data_root = f'/data/qingyuyang/ESM4SL/data/ryf/{cell_line}/{scene}'  # NOTE: change root path

    table_path = f'{output_root}/{cell_line}/{scene}'
    os.makedirs(table_path, exist_ok=True)
    metric_df = pd.DataFrame(0, index=range(6), columns=column_names)

    for test_fold in range(5):
        newest_version = len(os.listdir(f'{table_path}/fold_{test_fold}/csv_log')) - 1
        logit = pd.read_csv(f'{table_path}/fold_{test_fold}/csv_log/version_{newest_version}/test_logits.csv')
        gt = pd.read_csv(f'{data_root}/sl_test_{test_fold}.csv')
        combined_df = pd.merge(logit, gt, on=['0', '1'])
        combined_df.rename({'probs': 'pred_score', '0': 'gene1', '1': 'gene2', '2': 'label'}, axis=1, inplace=True)
        metrics = evaluate_performance(combined_df)
        for key in column_names:
            metric_df.at[test_fold, key] = metrics[key]

    mean_values = metric_df.iloc[0:5].mean()
    std_values = metric_df.iloc[0:5].std()
    new_row = [f"{mean:.4f} ({std:.4f})" for mean, std in zip(mean_values, std_values)]
    metric_df.iloc[5] = new_row
    metric_df.to_csv(f'{table_path}/{scene}.csv', index=False)
