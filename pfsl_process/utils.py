from __future__ import annotations
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.sparse as sp
from scipy.sparse import csr_matrix, triu


############################# General Settings #############################
root = '/data/qingyuyang/ESM4SL/data'

with open(f'{root}/mapping/name2id.pkl', 'rb') as f1:
    name2id = pickle.load(f1)  # Len = 27671. key: gene name, value: gene id, both str.
gid2name = {gid: name for name, gid in name2id.items()}

with open(f'{root}/mapping/gid2uid.pkl', 'rb') as f2:
    gid2uid = pickle.load(f2)  # Len = 9845. key: entrez id, value: unified id in SLBench, both int.


############################# SLKB Cleaning and Preprocessing #############################
def count_pn_ratio(df: pd.DataFrame) -> tuple[int, int, int, float]:
    total = len(df)
    pos = len(df[df['2'] == 1].index.tolist())
    neg = len(df[df['2'] == 0].index.tolist())
    if pos != 0:
        ratio = round(neg / pos, 4)
    else:
        ratio = None
    return total, pos, neg, ratio


def select_cl_from_slkb(df: pd.DataFrame, cl: str) -> pd.DataFrame:
    cl_df = df[df['3'] == cl]
    cl_df.reset_index(drop=True, inplace=True)
    cl_df.drop(['3'], axis=1, inplace=True)
    return cl_df


def remove_rows_with_condition(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df[df.duplicated(subset=['0', '1', '3'], keep=False)]
    for _, group in duplicates.groupby(['0', '1', '3']):
        assert len(group['2'].unique()) == len(group), f"Pair ({group.iloc[0]['0']}, {group.iloc[0]['1']}) have duplicate same labels!"
        df.drop(group.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['g1', 'g2', 'label', 'cell_line']]
    df.columns = ['0', '1', '2', '3']
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df


def df_gene2id(df: pd.DataFrame) -> pd.DataFrame:
    df['0'] = df['0'].map(name2id)
    df['1'] = df['1'].map(name2id)
    df = df.dropna()  # NOTE: some gene may not have id mappings
    df.reset_index(drop=True, inplace=True)
    df['0'] = df['0'].astype(int)
    df['1'] = df['1'].astype(int)
    return df


def filter_baseline_genes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['0'].isin(gid2uid.keys()) & df['1'].isin(gid2uid.keys())]
    df.reset_index(drop=True, inplace=True)
    return df


def count_gene_freq(df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, int]]:
    gene_list = list(df['0']) + list(df['1'])
    gene_set = list(set(gene_list))

    gene_count = {}
    for gene in gene_set:
        gene_count[gene] = gene_list.count(gene)
    return gene_list, gene_set, gene_count


def visualize_gene_freq(gene_count: dict, max_gene_show: int = 200, show_yval_step: int = 10, show_xlabel_step: int = 5, 
                        plot: bool = False, save_path: Path | None = None, title: str | None = None):
    gene_num = len(gene_count)
    jump = int(gene_num / max_gene_show) if gene_num >= max_gene_show else 1
    sorted_list = sorted(gene_count.items(), key=lambda item: item[1], reverse=True)[::jump]
    keys, values = zip(*sorted_list)
    keys = [gid2name[str(i)] for i in keys]

    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    bars = ax.bar(keys, values)
    for i, bar in enumerate(bars):
        if i % show_yval_step == 0:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
    if title:
        plt.title(f"{title}")
    plt.xlabel('Genes')
    plt.ylabel('Number of appearances in SL pairs')
    plt.xticks(np.arange(len(keys))[::show_xlabel_step], keys[::show_xlabel_step], rotation=90)
    if save_path:
        plt.savefig(f"{save_path}/{title}.jpg")
    if not plot:
        plt.clf()
        plt.close(fig)
    else:
        plt.show()


def are_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return (not df1.duplicated().any()) and (not df1.duplicated().any()) and len(df1) == len(df2) \
            and set(tuple(x) for x in df1.values) == set(tuple(x) for x in df2.values)


def are_dataframes_subset(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return (not df1.duplicated().any()) and (not df2.duplicated().any()) \
            and set(tuple(x) for x in df1.values) <= set(tuple(x) for x in df2.values)


def check_C2_property(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train_genes = list(set(train_df['0']) | set(train_df['1']))
    test_gene_pairs = list(zip(list(test_df['0']), list(test_df['1'])))

    for g1, g2 in test_gene_pairs:
        g1_freq = train_genes.count(g1)
        g2_freq = train_genes.count(g2)
        if not ((g1_freq > 0 and g2_freq == 0) or (g1_freq == 0 and g2_freq > 0)):
            index_names = test_df[(test_df['0'] == g1) & (test_df['1'] == g2)].index
            test_df.drop(index_names, inplace=True)

    return test_df


def check_C3_property(test_df: pd.DataFrame, train_df: pd.DataFrame) -> bool: #pd.DataFrame:
    train_genes = list(set(train_df['0']) | set(train_df['1']))
    test_gene_pairs = list(zip(list(test_df['0']), list(test_df['1'])))

    for g1, g2 in test_gene_pairs:
        g1_freq = train_genes.count(g1)
        g2_freq = train_genes.count(g2)
        if not (g1_freq == 0 and g2_freq == 0):
            return False

    return True  #test_df


def numpy2df(arr: np.ndarray):
    return pd.DataFrame(arr, columns=['0', '1', '2'])


def get_pairs(sl_pairs: np.ndarray[int], train_genes: np.ndarray[int], test_genes: np.ndarray[int], type: int) -> list[list[int]]:
    pairs_with_genes = []
    for pair in sl_pairs:
        if type == 1:
            if pair[0] in train_genes and pair[1] in train_genes:
                pairs_with_genes.append(list(pair))
        elif type == 2:
            if (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
                pairs_with_genes.append(list(pair))
        elif type == 3:
            if pair[0] in test_genes and pair[1] in test_genes:
                pairs_with_genes.append(list(pair))
    return pairs_with_genes


class SpecificSplit():
    def __init__(self, df: pd.DataFrame, save_path: Path, fold_num: int = 5, valid_ratio: int = 5, random_seed: int = 43) -> None:
        self.df = df
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.fold_num = fold_num
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed

    def C1(self) -> None:  #df: pd.DataFrame, save_path: Path, fold_num: int = 5, valid_ratio: int = 5
        C1_save_path = f'{self.save_path}/C1'
        os.makedirs(C1_save_path, exist_ok=True)

        sl_np_x = self.df[['0', '1']].to_numpy()
        sl_np_y = self.df['2'].to_numpy()

        # index is the fold
        index = 0
        kf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for train_index, test_index in kf.split(sl_np_x, sl_np_y):
            train_x = sl_np_x[train_index]
            train_y = sl_np_y[train_index]

            kf_train_valid = StratifiedKFold(n_splits=self.valid_ratio, shuffle=True, random_state=self.random_seed)  # keep percentage of each class
            for train_train_index, train_valid_index in kf_train_valid.split(train_x, train_y):
                train_train_x = train_x[train_train_index]
                train_train_y = train_y[train_train_index].reshape(-1, 1)

                train_valid_x = train_x[train_valid_index]
                train_valid_y = train_y[train_valid_index].reshape(-1, 1)

                train_data = np.concatenate((train_train_x, train_train_y), axis=1)
                valid_data = np.concatenate((train_valid_x, train_valid_y), axis=1)
                break

            test_x = sl_np_x[test_index]
            test_y = sl_np_y[test_index].reshape(-1, 1)
            test_data = np.concatenate((test_x, test_y), axis=1)

            train_data = numpy2df(train_data)
            valid_data = numpy2df(valid_data)
            test_data = numpy2df(test_data)
            assert are_dataframes_equal(pd.concat([train_data, valid_data, test_data], ignore_index=True), self.df)

            train_data.to_csv(f'{C1_save_path}/sl_train_{index}.csv', index=False)
            valid_data.to_csv(f'{C1_save_path}/sl_val_{index}.csv', index=False)
            test_data.to_csv(f'{C1_save_path}/sl_test_{index}.csv', index=False)
            index += 1

    def C2(self) -> None:  #df: pd.DataFrame, save_path: Path, fold_num: int = 5, valid_ratio: int = 5
        C2_save_path = f'{self.save_path}/C2'
        os.makedirs(C2_save_path, exist_ok=True)

        genes = np.array(list(set(self.df['0']) | set(self.df['1'])))
        sl_pairs = self.df.to_numpy()

        index = 0
        kf = KFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for train_index, test_index in kf.split(genes):
            train_genes = genes[train_index]
            test_genes = genes[test_index]

            # split train/valid from train gene
            kf_train_valid = KFold(n_splits=self.valid_ratio, shuffle=True, random_state=self.random_seed)
            for train_train_index, train_valid_index in kf_train_valid.split(train_genes):
                train_train_genes = train_genes[train_train_index]
                train_valid_genes = train_genes[train_valid_index]
                break

            train_data = get_pairs(sl_pairs, train_train_genes, test_genes=None, type=1)
            valid_c2_data = get_pairs(sl_pairs, train_train_genes, test_genes=train_valid_genes, type=2)
            test_c2_data = get_pairs(sl_pairs, train_train_genes, test_genes, type=2)

            train_data = numpy2df(train_data)
            valid_c2_data = numpy2df(valid_c2_data)
            test_c2_data = numpy2df(test_c2_data)
            assert are_dataframes_subset(pd.concat([train_data, valid_c2_data, test_c2_data], ignore_index=True), self.df)
            test_c2_data = check_C2_property(test_c2_data, train_data)  #pd.concat([train_data, valid_c2_data], ignore_index=True)
            valid_c2_data = check_C2_property(valid_c2_data, train_data)

            train_data.to_csv(f'{C2_save_path}/sl_train_{index}.csv', index=False)
            valid_c2_data.to_csv(f'{C2_save_path}/sl_val_{index}.csv', index=False)
            test_c2_data.to_csv(f'{C2_save_path}/sl_test_{index}.csv', index=False)
            index += 1

    def tail_train_test_split(self, df: pd.DataFrame, save_path: Path, test_rate: float = 0.2, min_tail_rate: float = 0.02, fold_num: int = 5) -> pd.DataFrame | None:
        _, _, cl_count = count_gene_freq(df)
        cl_tail_nodes = [gene for gene, freq in cl_count.items() if freq == 1]
        tail_df = df[df['0'].isin(cl_tail_nodes) & df['1'].isin(cl_tail_nodes)]
        tail_df = tail_df.sample(frac=1)

        if len(tail_df) <= int(min_tail_rate * len(df)):
            print('This cell line is unsuitable for long-tail scene!')
            os.removedirs(save_path)
            return None

        target_len = int(test_rate * len(df))
        test_df = tail_df[:target_len] if len(tail_df) > target_len else tail_df  # select some tails to form test set

        train_df = df.merge(test_df, how='left', indicator=True)  # train df = whole df - test df
        train_df = train_df[train_df['_merge'] == 'left_only'][['0', '1', '2']]  #, '3'

        assert are_dataframes_equal(pd.concat([train_df, test_df], ignore_index=True), df)

        for fold in range(fold_num):
            test_df.to_csv(f'{save_path}/sl_test_{fold}.csv', index=False)
        return train_df

    def Tail(self) -> None:  #df: pd.DataFrame, save_path: Path, fold_num: int = 5
        tail_save_path = f'{self.save_path}/Tail'
        os.makedirs(tail_save_path, exist_ok=True)

        df = self.tail_train_test_split(self.df, tail_save_path, fold_num=self.fold_num)
        if df is None:
            return None

        sl_pairs = df.to_numpy()
        train_genes = np.array(list(set(df['0']) | set(df['1'])))

        kf_train_valid = KFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for fold, (train_train_index, train_valid_index) in enumerate(kf_train_valid.split(train_genes)):
            train_train_genes = train_genes[train_train_index]
            train_valid_genes = train_genes[train_valid_index]

            train_pairs = get_pairs(sl_pairs, train_train_genes, test_genes=None, type=1)
            valid_pairs = get_pairs(sl_pairs, train_train_genes, test_genes=train_valid_genes, type=3)

            train_df = numpy2df(train_pairs)
            val_df = numpy2df(valid_pairs)

            assert are_dataframes_subset(pd.concat([train_df, val_df], ignore_index=True), df)
            assert check_C3_property(val_df, train_df)

            train_df.to_csv(f'{tail_save_path}/sl_train_{fold}.csv', index=False)
            val_df.to_csv(f'{tail_save_path}/sl_val_{fold}.csv', index=False)

    def cell_line_specific_all(self):
        self.C1()
        self.C2()
        self.Tail()


class TransferSplit():
    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                 save_path: Path, fold_num: int = 5, valid_ratio: int = 5, random_seed: int = 43) -> None:
        self.source_df = source_df
        self.target_df = target_df
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.fold_num = fold_num
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed

    def Transfer(self) -> None:  #df: pd.DataFrame, save_path: Path, fold_num: int = 5, valid_ratio: int = 5
        sl_np_x = self.source_df[['0', '1']].to_numpy()
        sl_np_y = self.source_df['2'].to_numpy()

        # index is the fold
        index = 0
        kf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for train_index, test_index in kf.split(sl_np_x, sl_np_y):
            train_x = sl_np_x[train_index]
            train_y = sl_np_y[train_index].reshape(-1, 1)
            train_data = np.concatenate((train_x, train_y), axis=1)

            test_x = sl_np_x[test_index]
            test_y = sl_np_y[test_index].reshape(-1, 1)
            test_data = np.concatenate((test_x, test_y), axis=1)

            train_data = numpy2df(train_data)
            test_data = numpy2df(test_data)
            assert are_dataframes_equal(pd.concat([train_data, test_data], ignore_index=True), self.df)

            train_data.to_csv(f'{self.save_path}/sl_train_{index}.csv', index=False)
            test_data.to_csv(f'{self.save_path}/sl_val_{index}.csv', index=False)
            self.target_df.to_csv(f'{self.save_path}/sl_test_{index}.csv', index=False)
            index += 1


def count_specific_statistics(table_save_path: Path) -> pd.DataFrame:
    column_names = ['cell_line', 'scene', 'fold', 'set', '# pairs', '# pos', '# neg', 'n/p ratio', '# genes', '# unique genes']
    datasets_stat = pd.DataFrame(columns=column_names)

    cell_lines = [i for i in os.listdir(table_save_path) if not i.endswith('.csv')]

    for cl, scene in tqdm(itertools.product(cell_lines, ['C1', 'C2', 'Tail'])):
        path = f'{table_save_path}/{cl}/{scene}'
        if not os.path.exists(path) or not len(os.listdir(path)):
            continue
        for fold in range(5):
            train_df = pd.read_csv(f'{path}/sl_train_{fold}.csv')
            total, pos, neg, ratio = count_pn_ratio(train_df)
            gene_list, gene_set, gene_count = count_gene_freq(train_df)
            unique_gene_num = list(gene_count.values()).count(1)
            datasets_stat.loc[len(datasets_stat)] = [cl, scene, fold, 0, total, pos, neg, ratio, len(gene_set), unique_gene_num]

            val_df = pd.read_csv(f'{path}/sl_val_{fold}.csv')
            total, pos, neg, ratio = count_pn_ratio(val_df)
            gene_list, gene_set, gene_count = count_gene_freq(val_df)
            unique_gene_num = list(gene_count.values()).count(1)
            datasets_stat.loc[len(datasets_stat)] = [cl, scene, fold, 1, total, pos, neg, ratio, len(gene_set), unique_gene_num]

            test_df = pd.read_csv(f'{path}/sl_test_{fold}.csv')
            total, pos, neg, ratio = count_pn_ratio(test_df)
            gene_list, gene_set, gene_count = count_gene_freq(test_df)
            unique_gene_num = list(gene_count.values()).count(1)
            datasets_stat.loc[len(datasets_stat)] = [cl, scene, fold, 2, total, pos, neg, ratio, len(gene_set), unique_gene_num]

    datasets_stat.to_csv(f'{table_save_path}/datasets_stat.csv', index=False)
    return datasets_stat


def form_id_seq_list(gids: set[int], mapping: pd.DataFrame, max_len: int = 2000) -> list[tuple[int, str]]:
    id_seq = []
    for gid in gids:
        seq = mapping[mapping['From'] == gid]['Sequence'].values[0]
        if len(seq) > max_len:
            seq = seq[:max_len]
        id_seq.append((gid, seq))
    return id_seq


############################# Baseline Preprocessing #############################
def df_gid2uid(df: pd.DataFrame) -> pd.DataFrame:
    df['0'] = df['0'].map(gid2uid)
    df['1'] = df['1'].map(gid2uid)
    df = df.dropna()  # NOTE: some gene may not have mappings
    df['0'] = df['0'].astype(int)
    df['1'] = df['1'].astype(int)
    return df


def gid2uid_all(data_path: Path, save_path: Path, fold: int = 5) -> None:
    os.makedirs(save_path, exist_ok=True)

    for i in range(fold):
        train_df_gid = pd.read_csv(f'{data_path}/sl_train_{i}.csv')
        train_df_uid = df_gid2uid(train_df_gid)
        assert len(train_df_uid) == len(train_df_gid)
        train_df_uid.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)

        val_df_gid = pd.read_csv(f'{data_path}/sl_val_{i}.csv')
        val_df_uid = df_gid2uid(val_df_gid)
        assert len(val_df_uid) == len(val_df_gid)
        val_df_uid.to_csv(f'{save_path}/sl_val_{i}.csv', index=False)

        test_df_gid = pd.read_csv(f'{data_path}/sl_test_{i}.csv')
        test_df_uid = df_gid2uid(test_df_gid)
        assert len(test_df_uid) == len(test_df_gid)
        test_df_uid.to_csv(f'{save_path}/sl_test_{i}.csv', index=False)


def gid2uid_all_onetoone(source_cl_root: Path, original_cl_root: Path, cl: str, save_root: Path, fold: int = 5) -> None:
    save_path = f'{save_root}/{cl}'
    os.makedirs(save_path, exist_ok=True)

    for i in range(fold):
        train_df_gid = pd.read_csv(f'{source_cl_root}/{cl}/sl_train_{i}.csv')
        train_df_uid = df_gid2uid(train_df_gid)
        assert len(train_df_uid) == len(train_df_gid)
        train_df_uid.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)

        val_df_gid = pd.read_csv(f'{source_cl_root}/{cl}/sl_val_{i}.csv')
        val_df_uid = df_gid2uid(val_df_gid)
        assert len(val_df_uid) == len(val_df_gid)
        val_df_uid.to_csv(f'{save_path}/sl_val_{i}.csv', index=False)

    test_df_gid = pd.read_csv(f'{original_cl_root}/{cl}.csv')
    test_df_uid = df_gid2uid(test_df_gid)
    assert len(test_df_uid) == len(test_df_gid)
    test_df_uid.to_csv(f'{save_root}/{cl}.csv', index=False)


def construct_data_npy(data_path: Path, save_path: Path, fold: int = 5, have_val: bool = True) -> None:
    sl_pos_train = []
    sl_neg_train = []
    if have_val:
        sl_pos_val = []
        sl_neg_val = []
    sl_pos_test = []
    sl_neg_test = []

    for i in range(fold):
        sl_data_train = pd.read_csv(os.path.join(data_path, f'sl_train_{i}.csv'))
        if have_val:
            sl_data_val = pd.read_csv(os.path.join(data_path, f'sl_val_{i}.csv'))
            sl_data_test = pd.read_csv(os.path.join(data_path, f'sl_test_{i}.csv'))
        else:
            sl_data_test = pd.read_csv(os.path.join(data_path, f'sl_val_{i}.csv'))  # NOTE: Use val as test set

        sl_data_pos_train = sl_data_train[['0', '1']][sl_data_train['2'] == 1].to_numpy()
        sl_data_neg_train = sl_data_train[['0', '1']][sl_data_train['2'] == 0].to_numpy()
        if have_val:
            sl_data_pos_val = sl_data_val[['0', '1']][sl_data_val['2'] == 1].to_numpy()
            sl_data_neg_val = sl_data_val[['0', '1']][sl_data_val['2'] == 0].to_numpy()
        sl_data_pos_test = sl_data_test[['0', '1']][sl_data_test['2'] == 1].to_numpy()
        sl_data_neg_test = sl_data_test[['0', '1']][sl_data_test['2'] == 0].to_numpy()

        sl_pos_train.append(sl_data_pos_train)
        sl_neg_train.append(sl_data_neg_train)
        if have_val:
            sl_pos_val.append(sl_data_pos_val)
            sl_neg_val.append(sl_data_neg_val)
        sl_pos_test.append(sl_data_pos_test)
        sl_neg_test.append(sl_data_neg_test)

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'pos_train.npy'), sl_pos_train)
    np.save(os.path.join(save_path, 'neg_train.npy'), sl_neg_train)
    if have_val:
        np.save(os.path.join(save_path, 'pos_val.npy'), sl_pos_val)
        np.save(os.path.join(save_path, 'neg_val.npy'), sl_neg_val)
    np.save(os.path.join(save_path, 'pos_test.npy'), sl_pos_test)
    np.save(os.path.join(save_path, 'neg_test.npy'), sl_neg_test)


def extract_sparse_mat_from_df(df_path: Path) -> tuple[csr_matrix, csr_matrix]:
    df = pd.read_csv(df_path)
    pos = np.array(df[df['2'] == 1])
    neg = np.array(df[df['2'] == 0])
    len_ = len(pos)
    len_neg = len(neg)
    sparse_matrix_pos = csr_matrix((np.ones(len_), (pos[:, 0], pos[:, 1])), shape=(9845, 9845)) 
    sparse_matrix_neg = csr_matrix((np.ones(len_neg), (neg[:, 0], neg[:, 1])), shape=(9845, 9845))
    return sparse_matrix_pos, sparse_matrix_neg


def construct_data_ptgnn(data_path: Path, save_path: Path = None, fold: int = 5, have_val: bool = True) -> None:      
    graph_train_pos_kfold = []
    graph_train_neg_kfold = []
    if have_val:
        graph_val_pos_kfold = []
        graph_val_neg_kfold = []
    graph_test_pos_kfold = []
    graph_test_neg_kfold = []

    for i in range(fold):
        sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_train_{i}.csv'))
        graph_train_pos_kfold.append(sparse_matrix_pos)
        graph_train_neg_kfold.append(sparse_matrix_neg)

        if have_val:
            sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_val_{i}.csv'))
            graph_val_pos_kfold.append(sparse_matrix_pos)
            graph_val_neg_kfold.append(sparse_matrix_neg)

            sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_test_{i}.csv'))
            graph_test_pos_kfold.append(sparse_matrix_pos)
            graph_test_neg_kfold.append(sparse_matrix_neg)
        else:
            sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_val_{i}.csv'))  # NOTE: Use val as test set
            graph_test_pos_kfold.append(sparse_matrix_pos)
            graph_test_neg_kfold.append(sparse_matrix_neg)

    if have_val:
        pos_graph = [graph_train_pos_kfold, graph_val_pos_kfold, graph_test_pos_kfold]
        neg_graph = [graph_train_neg_kfold, graph_val_neg_kfold, graph_test_neg_kfold]
    else:
        pos_graph = [graph_train_pos_kfold, graph_test_pos_kfold]
        neg_graph = [graph_train_neg_kfold, graph_test_neg_kfold]

    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/pos_graph.npy", pos_graph)
    np.save(f"{save_path}/neg_graph.npy", neg_graph)
