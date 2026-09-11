import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import scale

GENE1_COLNAME = 'gene1'
GENE2_COLNAME = 'gene2'

CL_FEAT_PATH = '/data/qingyuyang/SL/MVGCNiSL/data'


def load_graph_data(graph_type):
    if graph_type == "PPI-genetic" or graph_type == "PPI-physical":
        data = pd.read_csv(f"{CL_FEAT_PATH}/BIOGRID-9606.csv", index_col=0)

        if graph_type == "PPI-physical":
            data = data[data["Experimental System Type"] == "physical"]
        else:
            data = data[data["Experimental System Type"] != "physical"]
        print("Number of edges of {}: {}".format(graph_type, data.shape[0]))

        data = data[["Official Symbol Interactor A", "Official Symbol Interactor B"]]
        data.rename(columns={"Official Symbol Interactor A": GENE1_COLNAME, "Official Symbol Interactor B": GENE2_COLNAME}, inplace=True)

        # make it indirected graph
        data_dup = data.reindex(columns=[GENE2_COLNAME, GENE1_COLNAME])
        data_dup.columns = [GENE1_COLNAME, GENE2_COLNAME]
        data = data._append(data_dup)
    elif graph_type == "co-exp" or graph_type == "co-ess":
        if graph_type == "co-exp":
            data = pd.read_csv(f"{CL_FEAT_PATH}/coexpression_exp_0.5.csv")
        elif graph_type == "co-ess":
            data = pd.read_csv(f"{CL_FEAT_PATH}/coexpression_ess_0.2.csv")

        # make it indirected graph
        data_dup = data.reindex(columns=[GENE2_COLNAME, GENE1_COLNAME])
        data_dup.columns = [GENE1_COLNAME, GENE2_COLNAME]
        data = data._append(data_dup)

    return data


def choose_node_attribute(gene_mapping, cell_name):  # attr, , graph_data
    num_nodes = len(gene_mapping)
    
    feat_list = ["exp", "mut", "cnv", "ess"]
    dict_list = []
    for feat in feat_list:
        temp_df = pd.read_table("{}/cellline_feats/{}_{}.txt".format(CL_FEAT_PATH, cell_name, feat), names=["gene", "value"], sep=" ")
        # filter genes
        temp_df = temp_df[temp_df["gene"].isin(list(gene_mapping.keys()))]
        temp_dict = dict(zip(temp_df["gene"].values, temp_df["value"].values))
        dict_list.append(temp_dict)

    x = np.zeros((num_nodes, len(feat_list)))
    for col_idx, feat_dict in enumerate(dict_list):
        for key, value in feat_dict.items():
            row_idx = gene_mapping[key]
            x[row_idx, col_idx] = value
    
    # standardize features
    x = scale(x)
    return x


def merge_and_mapping(SL_data, graph_data_list, SL_genes):
    # use the union of SL genes and graph genes as all genes
    temp_concat_graph_data = pd.concat(graph_data_list)
    graph_genes = set(temp_concat_graph_data[GENE1_COLNAME].unique()) | set(temp_concat_graph_data[GENE2_COLNAME].unique())
    print("# graph genes:", len(graph_genes))
    all_genes = sorted(list(set(SL_genes) | graph_genes))
    print('len(SL_genes):', len(SL_genes))

    gene_mapping = dict(zip(all_genes, range(len(all_genes))))
    print('len(gene_mapping):', len(gene_mapping))

    # converting gene names to id
    # iterating over all graph types
    for i in range(len(graph_data_list)):
        graph_data_list[i][GENE1_COLNAME] = graph_data_list[i][GENE1_COLNAME].apply(lambda x: gene_mapping[x])
        graph_data_list[i][GENE2_COLNAME] = graph_data_list[i][GENE2_COLNAME].apply(lambda x: gene_mapping[x])

    SL_data[GENE1_COLNAME] = SL_data[GENE1_COLNAME].apply(lambda x: gene_mapping[x])
    SL_data[GENE2_COLNAME] = SL_data[GENE2_COLNAME].apply(lambda x: gene_mapping[x])

    return SL_data, graph_data_list, gene_mapping


def generate_feature(cell_name, data_path, fold, graph_input=["PPI-physical", "co-exp", "co-ess"]): #"PPI-genetic", 
    # load data
    SL_data_train = pd.read_csv(f"{data_path}/sl_train_{fold}.csv")
    SL_data_val = pd.read_csv(f"{data_path}/sl_val_{fold}.csv")
    SL_data_test = pd.read_csv(f"{data_path}/sl_test_{fold}.csv")
    SL_data = pd.concat([SL_data_train, SL_data_val, SL_data_test], ignore_index=True)
    SL_genes = list(set(SL_data[GENE1_COLNAME]) | set(SL_data[GENE2_COLNAME]))
    # print(type(SL_genes[0]))

    # load graph data
    graph_data_list = []
    # get cell-specific expression values to filter unexpressed genes in the network
    # remove genes whose expression raw count == 0
    if cell_name != "synlethdb":
        exp_df = pd.read_table("{}/cellline_feats/{}_exp.txt".format(CL_FEAT_PATH, cell_name), names=["gene", "value"], sep=" ")
        exp_df_filtered = exp_df[exp_df["value"] > 0]
        kept_genes = exp_df_filtered["gene"].values

    for graph_type in graph_input:
        # if graph_type == "PPI-genetic":
        #     graph_data = load_graph_data(graph_type)
        #     # removing edges already in the SL data
        #     SL_pos = SL_data[SL_data["label"] == True]
        #     SL_pos_list = sorted(
        #         [tuple(r) for r in SL_pos[[GENE1_COLNAME, GENE2_COLNAME]].to_numpy()]
        #         + [tuple(r) for r in SL_pos[[GENE2_COLNAME, GENE1_COLNAME]].to_numpy()]
        #     )
        #     graph_list = [tuple(r) for r in graph_data[[GENE1_COLNAME, GENE2_COLNAME]].to_numpy()]
        #     left = list(set(graph_list) - set(SL_pos_list))
        #     graph_data = pd.DataFrame(left, columns=[GENE1_COLNAME, GENE2_COLNAME])
        #     if cell_name != "synlethdb":
        #         graph_data = graph_data[(graph_data[GENE1_COLNAME].isin(kept_genes)) & (graph_data[GENE2_COLNAME].isin(kept_genes))]
        # else:
        graph_data = load_graph_data(graph_type)
        if cell_name != "synlethdb":
            graph_data = graph_data[(graph_data[GENE1_COLNAME].isin(kept_genes)) & (graph_data[GENE2_COLNAME].isin(kept_genes))]
            # graph_data = graph_data[(graph_data[GENE1_COLNAME].isin(SL_genes)) & (graph_data[GENE2_COLNAME].isin(SL_genes))]

        graph_data_list.append(graph_data)

    # merge, filter and mapping
    SL_data, graph_data_list, gene_mapping = merge_and_mapping(SL_data, graph_data_list, SL_genes)

    # generate node features
    x = choose_node_attribute(gene_mapping, cell_name) #attr, , graph_data_list

    # generate torch data
    data_x = torch.tensor(x, dtype=torch.float)

    data_edge_index_list = []
    for graph_data in graph_data_list:
        temp_edge_index = torch.tensor([graph_data[GENE1_COLNAME].values, graph_data[GENE2_COLNAME].values], dtype=torch.long)
        data_edge_index_list.append(temp_edge_index)

    data = Data(x=data_x, edge_index_list=data_edge_index_list)

    SL_data_train[GENE1_COLNAME] = SL_data_train[GENE1_COLNAME].map(gene_mapping)
    SL_data_train[GENE2_COLNAME] = SL_data_train[GENE2_COLNAME].map(gene_mapping)
    SL_data_val[GENE1_COLNAME] = SL_data_val[GENE1_COLNAME].map(gene_mapping)
    SL_data_val[GENE2_COLNAME] = SL_data_val[GENE2_COLNAME].map(gene_mapping)
    SL_data_test[GENE1_COLNAME] = SL_data_test[GENE1_COLNAME].map(gene_mapping)
    SL_data_test[GENE2_COLNAME] = SL_data_test[GENE2_COLNAME].map(gene_mapping)

    return data, SL_data_train, SL_data_val, SL_data_test, gene_mapping
