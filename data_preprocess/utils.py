from __future__ import annotations
from collections import Counter
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

root = '/data/qingyuyang/SL/PFSL/data'

GENE1_COLNAME = 'gene1' #'gene1'
GENE2_COLNAME = 'gene2' #'gene2'
CLASSLABEL_COLNAME = 'label'
# REGLABEL_COLNAME = 'score'
CONTEXT_COLNAME = 'context'


############################# SLKB Cleaning and Preprocessing #############################

synlethdb = pd.read_csv(f'{root}/mapping/meta_table_9845.csv')

gn2gid = dict(zip(
    list(synlethdb['symbol']),
    [int(i) for i in list(synlethdb['entrez_id'])]
))
gid2gn = {gid: gn for gn, gid in gn2gid.items()}

gid2uid = dict(zip(
    [int(i) for i in list(synlethdb['entrez_id'])],
    [int(i) for i in list(synlethdb['unified_id'])]
))


def count_gene_num(df: pd.DataFrame) -> int:
    return len(set(df[GENE1_COLNAME]) | set(df[GENE2_COLNAME]))


def count_pn_ratio(df: pd.DataFrame) -> tuple[int, int, int, float]:
    total = len(df)
    pos = len(df[df[CLASSLABEL_COLNAME] == 1].index.tolist())
    neg = len(df[df[CLASSLABEL_COLNAME] == 0].index.tolist())
    if pos != 0:
        ratio = round(neg / pos, 4)
    else:
        ratio = None
    return total, pos, neg, ratio


def select_cl_from_slkb(df: pd.DataFrame, cl: str) -> pd.DataFrame:
    cl_df = df[df[CONTEXT_COLNAME] == cl]
    cl_df.reset_index(drop=True, inplace=True)
    # cl_df.drop([CONTEXT_COLNAME], axis=1, inplace=True)
    return cl_df


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Copy and order endpoints after gene-ID mapping; context stays part of identity."""
    columns = [GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME]
    if CONTEXT_COLNAME in df.columns:
        columns.append(CONTEXT_COLNAME)
    if df[columns].isna().any().any():
        raise ValueError('Pair identity, context and label must not contain missing values.')
    if not df[CLASSLABEL_COLNAME].isin([0, 1]).all():
        raise ValueError('Expected binary pair labels 0/1.')
    result = df.copy()
    result[[GENE1_COLNAME, GENE2_COLNAME]] = np.sort(
        result[[GENE1_COLNAME, GENE2_COLNAME]].to_numpy(), axis=1)
    return result


def _pair_columns(df: pd.DataFrame) -> list[str]:
    return [GENE1_COLNAME, GENE2_COLNAME] + ([CONTEXT_COLNAME] if CONTEXT_COLNAME in df.columns else [])


def remove_rows_with_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one observation per unordered pair/context; drop ALL conflicting labels.

    This preserves the original conflict-removal policy, including reverse pairs.
    It also works on single-cell tables without a context column, without mutation.
    """
    result = canonicalize_pairs(df)
    keys = _pair_columns(result)
    conflicts = result.groupby(keys)[CLASSLABEL_COLNAME].transform('nunique').gt(1)
    return result.loc[~conflicts].drop_duplicates(subset=keys).reset_index(drop=True)


def validate_clean_pairs(df: pd.DataFrame) -> None:
    """Reject dirty input before splitting; never repair each split independently."""
    canonical = canonicalize_pairs(df)
    if canonical.duplicated(subset=_pair_columns(canonical)).any():
        raise ValueError('Duplicate/conflicting unordered pair within context; clean the full table before splitting.')


def balance_binary_pairs(df: pd.DataFrame, random_seed: int = 43) -> pd.DataFrame:
    """Downsample whichever class is larger to create a true 1:1 cell table."""
    validate_clean_pairs(df)
    pos = df[df[CLASSLABEL_COLNAME] == 1]
    neg = df[df[CLASSLABEL_COLNAME] == 0]
    size = min(len(pos), len(neg))
    if size == 0:
        raise ValueError('A 1:1 table requires both classes; exclude single-class cells explicitly.')
    return pd.concat([part.sample(n=size, random_state=random_seed) for part in [pos, neg]],
                     ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)


def df_gene2id(df: pd.DataFrame) -> pd.DataFrame:
    df[GENE1_COLNAME] = df[GENE1_COLNAME].map(gn2gid)
    df[GENE2_COLNAME] = df[GENE2_COLNAME].map(gn2gid)
    df = df.dropna(ignore_index=True)  # NOTE: drop genes not in SynlethDB
    df[GENE1_COLNAME] = df[GENE1_COLNAME].astype(int)
    df[GENE2_COLNAME] = df[GENE2_COLNAME].astype(int)
    return df


def filter_baseline_genes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[GENE1_COLNAME].isin(gid2uid.keys()) & df[GENE2_COLNAME].isin(gid2uid.keys())]
    df.reset_index(drop=True, inplace=True)
    return df


def count_gene_freq(df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, int]]:
    gene_list = list(df[GENE1_COLNAME]) + list(df[GENE2_COLNAME])
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
    keys = [gid2gn[i] for i in keys]

    fig, ax = plt.subplots(figsize=(20, 8), dpi=300)
    bars = ax.bar(keys, values)
    for i, bar in enumerate(bars):
        if i % show_yval_step == 0:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
    if title:
        plt.title(title)
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
    if (not df1.columns.is_unique or not df2.columns.is_unique
            or set(df1.columns) != set(df2.columns)):
        return False
    aligned = df2.loc[:, df1.columns]
    return (not df1.duplicated().any()) and (not aligned.duplicated().any()) and len(df1) == len(aligned) \
            and set(df1.itertuples(index=False, name=None)) == set(aligned.itertuples(index=False, name=None))


def are_dataframes_subset(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    if (not df1.columns.is_unique or not df2.columns.is_unique
            or set(df1.columns) != set(df2.columns)):
        return False
    aligned = df2.loc[:, df1.columns]
    return (not df1.duplicated().any()) and (not aligned.duplicated().any()) \
            and set(df1.itertuples(index=False, name=None)) <= set(aligned.itertuples(index=False, name=None))


def check_C2_property(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    train_genes = list(set(train_df[GENE1_COLNAME]) | set(train_df[GENE2_COLNAME]))
    test_gene_pairs = list(zip(list(test_df[GENE1_COLNAME]), list(test_df[GENE2_COLNAME])))

    for g1, g2 in test_gene_pairs:
        g1_freq = train_genes.count(g1)
        g2_freq = train_genes.count(g2)
        if not ((g1_freq > 0 and g2_freq == 0) or (g1_freq == 0 and g2_freq > 0)):
            index_names = test_df[(test_df[GENE1_COLNAME] == g1) & (test_df[GENE2_COLNAME] == g2)].index
            test_df.drop(index_names, inplace=True)

    return test_df


def check_C3_property(test_df: pd.DataFrame, train_df: pd.DataFrame) -> bool: #pd.DataFrame:
    train_genes = list(set(train_df[GENE1_COLNAME]) | set(train_df[GENE2_COLNAME]))
    test_gene_pairs = list(zip(list(test_df[GENE1_COLNAME]), list(test_df[GENE2_COLNAME])))

    for g1, g2 in test_gene_pairs:
        g1_freq = train_genes.count(g1)
        g2_freq = train_genes.count(g2)
        if not (g1_freq == 0 and g2_freq == 0):
            return False

    return True  #test_df


def numpy2df(arr: np.ndarray, context: str):
    df = pd.DataFrame(arr, columns=[GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME])
    df[CONTEXT_COLNAME] = context
    return df


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
        validate_clean_pairs(df)

        self.df = df.copy()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.fold_num = fold_num
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed

        contexts = list(set(self.df[CONTEXT_COLNAME]))
        assert len(contexts) == 1
        self.context = contexts[0]

    def C1(self) -> None:
        C1_save_path = f'{self.save_path}/C1'
        os.makedirs(C1_save_path, exist_ok=True)

        sl_np_x = self.df[[GENE1_COLNAME, GENE2_COLNAME]].to_numpy(dtype=int)
        sl_np_y = self.df[CLASSLABEL_COLNAME].to_numpy(dtype=int)

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

            train_data = numpy2df(train_data, self.context)
            valid_data = numpy2df(valid_data, self.context)
            test_data = numpy2df(test_data, self.context)
            assert are_dataframes_equal(pd.concat([train_data, valid_data, test_data], ignore_index=True), self.df)

            train_data.to_csv(f'{C1_save_path}/sl_train_{index}.csv', index=False)
            valid_data.to_csv(f'{C1_save_path}/sl_val_{index}.csv', index=False)
            test_data.to_csv(f'{C1_save_path}/sl_test_{index}.csv', index=False)
            index += 1

    def C2(self) -> None:
        C2_save_path = f'{self.save_path}/C2'
        os.makedirs(C2_save_path, exist_ok=True)

        genes = np.array(list(set(self.df[GENE1_COLNAME]) | set(self.df[GENE2_COLNAME])))
        sl_pairs = self.df[[GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME]].to_numpy(dtype=int)

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

            train_data = numpy2df(train_data, self.context)
            valid_c2_data = numpy2df(valid_c2_data, self.context)
            test_c2_data = numpy2df(test_c2_data, self.context)
            assert are_dataframes_subset(pd.concat([train_data, valid_c2_data, test_c2_data], ignore_index=True), self.df)
            test_c2_data = check_C2_property(test_c2_data, train_data)  #pd.concat([train_data, valid_c2_data], ignore_index=True)
            valid_c2_data = check_C2_property(valid_c2_data, train_data)

            train_data.to_csv(f'{C2_save_path}/sl_train_{index}.csv', index=False)
            valid_c2_data.to_csv(f'{C2_save_path}/sl_val_{index}.csv', index=False)
            test_c2_data.to_csv(f'{C2_save_path}/sl_test_{index}.csv', index=False)
            index += 1
    
    def C3(self) -> None:
        C3_save_path = f'{self.save_path}/C3'
        os.makedirs(C3_save_path, exist_ok=True)

        genes = np.array(list(set(self.df[GENE1_COLNAME]) | set(self.df[GENE2_COLNAME])))
        sl_pairs = self.df[[GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME]].to_numpy(dtype=int)

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
            valid_c3_data = get_pairs(sl_pairs, train_train_genes, test_genes=train_valid_genes, type=3)
            test_c3_data = get_pairs(sl_pairs, train_train_genes, test_genes, type=3)

            train_data = numpy2df(train_data, self.context)
            valid_c3_data = numpy2df(valid_c3_data, self.context)
            test_c3_data = numpy2df(test_c3_data, self.context)
            assert are_dataframes_subset(pd.concat([train_data, valid_c3_data, test_c3_data], ignore_index=True), self.df)
            assert check_C3_property(test_c3_data, pd.concat([train_data, valid_c3_data], ignore_index=True))  #
            assert check_C3_property(valid_c3_data, train_data)

            train_data.to_csv(f'{C3_save_path}/sl_train_{index}.csv', index=False)
            valid_c3_data.to_csv(f'{C3_save_path}/sl_val_{index}.csv', index=False)
            test_c3_data.to_csv(f'{C3_save_path}/sl_test_{index}.csv', index=False)
            index += 1

    def tail_train_test_split(self, df: pd.DataFrame, save_path: Path, test_rate: float = 0.2, min_tail_rate: float = 0.02, fold_num: int = 5) -> pd.DataFrame | None:
        _, _, cl_count = count_gene_freq(df)
        cl_tail_nodes = [gene for gene, freq in cl_count.items() if freq == 1]
        tail_df = df[df[GENE1_COLNAME].isin(cl_tail_nodes) & df[GENE2_COLNAME].isin(cl_tail_nodes)]
        tail_df = tail_df.sample(frac=1, random_state=self.random_seed)

        if len(tail_df) <= int(min_tail_rate * len(df)):
            print('This cell line is unsuitable for long-tail scene!')
            if os.path.exists(save_path):
                os.removedirs(save_path)
            return None

        target_len = int(test_rate * len(df))
        test_df = tail_df[:target_len] if len(tail_df) > target_len else tail_df  # select some tails to form test set

        train_df = df.merge(test_df, how='left', indicator=True)  # train df = whole df - test df
        train_df = train_df.loc[train_df['_merge'] == 'left_only', df.columns]

        assert are_dataframes_equal(pd.concat([train_df, test_df], ignore_index=True), df)

        for fold in range(fold_num):
            test_df.to_csv(f'{save_path}/sl_test_{fold}.csv', index=False)
        return train_df

    def Tail(self) -> None:
        tail_save_path = f'{self.save_path}/Tail'
        os.makedirs(tail_save_path, exist_ok=True)

        df = self.tail_train_test_split(self.df, tail_save_path, fold_num=self.fold_num)
        if df is None:
            return None

        sl_pairs = df[[GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME]].to_numpy(dtype=int)
        train_genes = np.array(list(set(df[GENE1_COLNAME]) | set(df[GENE2_COLNAME])))

        kf_train_valid = KFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for fold, (train_train_index, train_valid_index) in enumerate(kf_train_valid.split(train_genes)):
            train_train_genes = train_genes[train_train_index]
            train_valid_genes = train_genes[train_valid_index]

            train_pairs = get_pairs(sl_pairs, train_train_genes, test_genes=None, type=1)
            valid_pairs = get_pairs(sl_pairs, train_train_genes, test_genes=train_valid_genes, type=3)

            train_df = numpy2df(train_pairs, self.context)
            val_df = numpy2df(valid_pairs, self.context)

            assert are_dataframes_subset(pd.concat([train_df, val_df], ignore_index=True), df)
            assert check_C3_property(val_df, train_df)

            train_df.to_csv(f'{tail_save_path}/sl_train_{fold}.csv', index=False)
            val_df.to_csv(f'{tail_save_path}/sl_val_{fold}.csv', index=False)

    def all_scenes(self):
        self.C1()
        self.C2()
        self.C3()
        self.Tail()


class OnetoOneSplit():
    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                 save_path: Path, fold_num: int = 5, random_seed: int = 43) -> None:
        validate_clean_pairs(source_df)
        validate_clean_pairs(target_df)
        self.source_df = source_df.copy()
        self.target_df = target_df.copy()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.fold_num = fold_num
        self.random_seed = random_seed

    def split(self) -> None:
        sl_np_x = self.source_df[[GENE1_COLNAME, GENE2_COLNAME]].to_numpy(dtype=int)
        sl_np_y = self.source_df[CLASSLABEL_COLNAME].to_numpy(dtype=int)

        kf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)

        for index, (train_index, valid_index) in enumerate(kf.split(sl_np_x, sl_np_y)):
            train_data = self.source_df.iloc[train_index].reset_index(drop=True)
            valid_data = self.source_df.iloc[valid_index].reset_index(drop=True)
            # assert are_dataframes_equal(
            #     pd.concat([train_data, valid_data], ignore_index=True),
            #     self.source_df,
            # )

            train_data.to_csv(f'{self.save_path}/sl_train_{index}.csv', index=False)
            valid_data.to_csv(f'{self.save_path}/sl_val_{index}.csv', index=False)
            self.target_df.to_csv(f'{self.save_path}/sl_test_{index}.csv', index=False)


class MultitoOneSplit(OnetoOneSplit):
    def __init__(self, source_dfs: dict[str, pd.DataFrame], target_df: pd.DataFrame,
                 target_context: str, save_path: Path, fold_num: int = 5,
                 random_seed: int = 43, balanced: bool = False) -> None:
        self.balanced = balanced
        if len(source_dfs) < 2:
            raise ValueError('MultitoOneSplit requires at least two source contexts.')
        if target_context in source_dfs:
            raise ValueError('The target context cannot also be a source context.')

        columns = [GENE1_COLNAME, GENE2_COLNAME, CLASSLABEL_COLNAME]
        source_parts = []
        for context, source_df in source_dfs.items():
            source_part = source_df[columns].copy()
            source_part[CONTEXT_COLNAME] = context
            source_parts.append(source_part)

        source_df = pd.concat(source_parts, ignore_index=True)
        target_df = target_df[columns].copy()
        target_df[CONTEXT_COLNAME] = target_context

        super().__init__(
            source_df=source_df,
            target_df=target_df,
            save_path=save_path,
            fold_num=fold_num,
            random_seed=random_seed,
        )

    def split(self) -> None:
        if not self.balanced:
            return super().split()
        # Match positive/negative fold sizes within each source context.
        # Unlike splitting the concatenated labels, this is exactly 1:1 even
        # when the number of positives is not divisible by fold_num.
        counts = self.target_df[CLASSLABEL_COLNAME].value_counts()
        if counts.get(0, 0) != counts.get(1, 0) or not counts.get(1, 0):
            raise ValueError('Balanced multi-to-one requires a balanced target table')
        trains = [[] for _ in range(self.fold_num)]
        vals = [[] for _ in range(self.fold_num)]
        for context, frame in self.source_df.groupby(CONTEXT_COLNAME, sort=True):
            positive = frame[frame[CLASSLABEL_COLNAME] == 1].reset_index(drop=True)
            negative = frame[frame[CLASSLABEL_COLNAME] == 0].sample(frac=1, random_state=self.random_seed + 1).reset_index(drop=True)
            if len(positive) != len(negative) or len(positive) < self.fold_num:
                raise ValueError(f'{context}: need balanced classes with at least fold_num pairs each')
            kf = KFold(n_splits=self.fold_num, shuffle=True, random_state=self.random_seed)
            for fold, (train, val) in enumerate(kf.split(positive)):
                trains[fold].extend([positive.iloc[train], negative.iloc[train]])
                vals[fold].extend([positive.iloc[val], negative.iloc[val]])
        for fold in range(self.fold_num):
            for split, parts in [('train', trains[fold]), ('val', vals[fold])]:
                frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=self.random_seed + fold)
                frame.to_csv(f'{self.save_path}/sl_{split}_{fold}.csv', index=False)
            self.target_df.to_csv(f'{self.save_path}/sl_test_{fold}.csv', index=False)


def count_specific_statistics(table_save_path: Path) -> pd.DataFrame:
    column_names = ['cell_line', 'scene', 'fold', 'set', '# pairs', '# pos', '# neg', 'n/p ratio', '# genes', '# unique genes']
    datasets_stat = pd.DataFrame(columns=column_names)

    cell_lines = [i for i in os.listdir(table_save_path) if not i.endswith('.csv')]

    for cl, scene in tqdm(itertools.product(cell_lines, ['C1', 'C2', 'C3', 'Tail'])):
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
    df[GENE1_COLNAME] = df[GENE1_COLNAME].map(gid2uid)
    df[GENE2_COLNAME] = df[GENE2_COLNAME].map(gid2uid)
    df = df.dropna()  # NOTE: some gene may not have mappings
    df[GENE1_COLNAME] = df[GENE1_COLNAME].astype(int)
    df[GENE2_COLNAME] = df[GENE2_COLNAME].astype(int)
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


# def gid2uid_all_onetoone(source_cl_root: Path, original_cl_root: Path, cl: str, save_root: Path, fold: int = 5) -> None:
#     save_path = f'{save_root}/{cl}'
#     os.makedirs(save_path, exist_ok=True)

#     for i in range(fold):
#         train_df_gid = pd.read_csv(f'{source_cl_root}/{cl}/sl_train_{i}.csv')
#         train_df_uid = df_gid2uid(train_df_gid)
#         assert len(train_df_uid) == len(train_df_gid)
#         train_df_uid.to_csv(f'{save_path}/sl_train_{i}.csv', index=False)

#         val_df_gid = pd.read_csv(f'{source_cl_root}/{cl}/sl_val_{i}.csv')
#         val_df_uid = df_gid2uid(val_df_gid)
#         assert len(val_df_uid) == len(val_df_gid)
#         val_df_uid.to_csv(f'{save_path}/sl_val_{i}.csv', index=False)

#     test_df_gid = pd.read_csv(f'{original_cl_root}/{cl}.csv')
#     test_df_uid = df_gid2uid(test_df_gid)
#     assert len(test_df_uid) == len(test_df_gid)
#     test_df_uid.to_csv(f'{save_root}/{cl}.csv', index=False)


def _save_pair_folds(path: Path, folds: list[np.ndarray]) -> None:
    # Unequal fold sizes need explicit object storage in modern NumPy.
    if len({array.shape for array in folds}) > 1:
        arrays = np.empty(len(folds), dtype=object)
        arrays[:] = folds
    else:
        arrays = np.asarray(folds)
    np.save(path, arrays)


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
            sl_data_test = pd.read_csv(os.path.join(data_path, f'sl_test_{i}.csv'))  # have_val only controls validation export

        sl_data_pos_train = sl_data_train[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_train[CLASSLABEL_COLNAME] == 1].to_numpy(dtype=int)
        sl_data_neg_train = sl_data_train[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_train[CLASSLABEL_COLNAME] == 0].to_numpy(dtype=int)
        if have_val:
            sl_data_pos_val = sl_data_val[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_val[CLASSLABEL_COLNAME] == 1].to_numpy(dtype=int)
            sl_data_neg_val = sl_data_val[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_val[CLASSLABEL_COLNAME] == 0].to_numpy(dtype=int)
        sl_data_pos_test = sl_data_test[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_test[CLASSLABEL_COLNAME] == 1].to_numpy(dtype=int)
        sl_data_neg_test = sl_data_test[[GENE1_COLNAME, GENE2_COLNAME]][sl_data_test[CLASSLABEL_COLNAME] == 0].to_numpy(dtype=int)

        sl_pos_train.append(sl_data_pos_train)
        sl_neg_train.append(sl_data_neg_train)
        if have_val:
            sl_pos_val.append(sl_data_pos_val)
            sl_neg_val.append(sl_data_neg_val)
        sl_pos_test.append(sl_data_pos_test)
        sl_neg_test.append(sl_data_neg_test)

    os.makedirs(save_path, exist_ok=True)
    _save_pair_folds(os.path.join(save_path, 'pos_train.npy'), sl_pos_train)
    _save_pair_folds(os.path.join(save_path, 'neg_train.npy'), sl_neg_train)
    if have_val:
        _save_pair_folds(os.path.join(save_path, 'pos_val.npy'), sl_pos_val)
        _save_pair_folds(os.path.join(save_path, 'neg_val.npy'), sl_neg_val)
    _save_pair_folds(os.path.join(save_path, 'pos_test.npy'), sl_pos_test)
    _save_pair_folds(os.path.join(save_path, 'neg_test.npy'), sl_neg_test)


def extract_sparse_mat_from_df(df_path: Path) -> tuple[csr_matrix, csr_matrix]:
    df = pd.read_csv(df_path)
    # Metadata such as context may precede the gene columns in exported CSVs.
    pair_columns = [GENE1_COLNAME, GENE2_COLNAME]
    pos = df.loc[df[CLASSLABEL_COLNAME] == 1, pair_columns].to_numpy(dtype=np.int64)
    neg = df.loc[df[CLASSLABEL_COLNAME] == 0, pair_columns].to_numpy(dtype=np.int64)
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
            sparse_matrix_pos, sparse_matrix_neg = extract_sparse_mat_from_df(os.path.join(data_path, f'sl_test_{i}.csv'))  # have_val only controls validation export
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
