import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


def load_graph_data(graph_type):
    if graph_type == 'PPI-genetic' or graph_type == 'PPI-physical':
        data = pd.read_csv("../data/BIOGRID-9606.csv", index_col=0)
        # all_genes = set(data['Official Symbol Interactor A'].unique()) | set(data['Official Symbol Interactor B'].unique())
        
        if graph_type == 'PPI-physical':
            data = data[data['Experimental System Type'] == 'physical']
        else:
            data = data[data['Experimental System Type'] != 'physical']
        print("Number of edges of {}: {}".format(graph_type, data.shape[0]))

        data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
        data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)

        # make it indirected graph
        data_dup = data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        data = data._append(data_dup)
    # elif graph_type == "pathway":
    #     data = pd.read_csv("../data/Opticon_networks.csv")
    #     data.rename(columns={"Regulator":"gene1", "Target gene":"gene2"}, inplace=True)
    elif graph_type == 'co-exp' or graph_type == 'co-ess':
        if graph_type == 'co-exp':
            data = pd.read_csv("../data/coexpression_exp_0.5.csv")
        elif graph_type == 'co-ess':
            data = pd.read_csv("../data/coexpression_ess_0.2.csv")

        # make it indirected graph
        data_dup = data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        data = data._append(data_dup)
    # elif graph_type == "random":
    #     data = pd.read_csv("../data/BIOGRID-9606.csv", index_col=0)
        
    #     data = data[data['Experimental System Type'] == 'physical']
    #     data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
    #     data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)
    #     all_genes = set(data['gene1'].unique()) | set(data['gene2'].unique())
    #     dict_mapping = dict(zip(range(len(all_genes)), all_genes))
        
    #     num_nodes = len(all_genes)
    #     num_edges = data.shape[0]
    #     p = 2*num_edges/(num_nodes*(num_nodes-1))
    #     # make it more sparse
    #     p = p/10.0
        
    #     G = fast_gnp_random_graph(num_nodes, p)
    #     data = nx.convert_matrix.to_pandas_edgelist(G,source='gene1',target='gene2')
    #     print("generated number of edges: {}".format(G.number_of_edges()))

    return data


def choose_node_attribute(attr, gene_mapping, cell_name, graph_data):
    num_nodes = len(gene_mapping)
    if attr == "identity":
        x = np.identity(num_nodes)
    elif attr == 'random':
        x = np.random.randn(num_nodes, 4)
    elif attr == 'raw_omics':
        feat_list = ['exp','mut','cnv','ess']
        dict_list = []
        for feat in feat_list:
            temp_df = pd.read_table("../data/cellline_feats/{}_{}.txt".format(cell_name,feat),
                                        names=['gene','value'], sep=' ')
            # filter genes
            temp_df = temp_df[temp_df['gene'].isin(list(gene_mapping.keys()))]
            temp_dict = dict(zip(temp_df['gene'].values, temp_df['value'].values))
            dict_list.append(temp_dict)
        
        x = np.zeros((num_nodes, len(feat_list)))
        for col_idx, feat_dict in enumerate(dict_list):
            for key, value in feat_dict.items():
                row_idx = gene_mapping[key]
                x[row_idx, col_idx] = value
        # standardize features
        x = scale(x)
    elif attr == "CCLE_ess" or attr == "CCLE_exp":
        dim = 64
        df = pd.read_csv("../data/CCLE/{}.csv".format(attr), index_col=0)
        df.fillna(0, inplace=True)
        df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
        
        df = df.T
        all_genes = list(df.index)

        # perform PCA for dimension reduction
        values = df.values
        pca = PCA(n_components=dim)
        pca.fit(values)
        embed = pca.transform(values)
        embed_df = pd.DataFrame(embed, index=all_genes)

        x = np.zeros((num_nodes, dim))
        for gene, idx in gene_mapping.items():
            if gene in all_genes:
                x[idx] = embed_df.loc[gene]
        x = scale(x)
    elif attr == 'node2vec':
        # need to first build a torch data
        data_x = torch.tensor(np.random.randn(num_nodes, 128), dtype=torch.float)

        # concat all types of graph data
        graph_data_overall = pd.concat(graph_data)
        data_edge_index = torch.tensor([graph_data_overall['gene1'].values, graph_data_overall['gene2'].values], dtype=torch.long)
        
        data = Data(x=data_x, edge_index=data_edge_index)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10, context_size=10, sparse=True).to(device)
        loader = model.loader(batch_size=512, shuffle=True, num_workers=28)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss/len(loader)
        
        for epoch in range(1, 51):
            loss = train()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        
        # get embeddings
        model.eval()
        x = model(torch.arange(data.num_nodes, device=device))
        print(x.size())
        x = x.cpu().detach().numpy()

    return x


def merge_and_mapping(SL_data, graph_data_list, SL_genes):
    # use the union of SL genes and graph genes as all genes
    temp_concat_graph_data = pd.concat(graph_data_list)
    graph_genes = set(temp_concat_graph_data['gene1'].unique()) | set(temp_concat_graph_data['gene2'].unique())
    print('# graph genes:', len(graph_genes))
    all_genes = sorted(list(set(SL_genes) | graph_genes))

    gene_mapping = dict(zip(all_genes, range(len(all_genes))))

    # converting gene names to id
    # iterating over all graph types
    for i in range(len(graph_data_list)):
        graph_data_list[i]['gene1'] = graph_data_list[i]['gene1'].apply(lambda x:gene_mapping[x])
        graph_data_list[i]['gene2'] = graph_data_list[i]['gene2'].apply(lambda x:gene_mapping[x])
    
    SL_data['gene1'] = SL_data['gene1'].apply(lambda x:gene_mapping[x])
    SL_data['gene2'] = SL_data['gene2'].apply(lambda x:gene_mapping[x])

    return SL_data, graph_data_list, gene_mapping


def generate_train_feature_self(cell_name, data_path, fold, graph_input, attr):
    # load data
    SL_data_train = pd.read_csv(f'{data_path}/sl_train_{fold}.csv')
    SL_data_val = pd.read_csv(f'{data_path}/sl_val_{fold}.csv')
    SL_data_test = pd.read_csv(f'{data_path}/sl_test_{fold}.csv')
    SL_data = pd.concat([SL_data_train, SL_data_val, SL_data_test], ignore_index=True)
    SL_genes = list(set(SL_data['gene1']) | set(SL_data['gene2']))
    
    # load graph data
    graph_data_list = []
    # get cell-specific expression values to filter unexpressed genes in the network
    # remove genes whose expression raw count == 0
    if cell_name != "synlethdb":
        exp_df = pd.read_table("../data/cellline_feats/{}_exp.txt".format(cell_name),names=['gene','value'], sep=' ')
        exp_df_filtered = exp_df[exp_df["value"]>0]
        kept_genes = exp_df_filtered["gene"].values

    for graph_type in graph_input:
        if graph_type == "PPI-genetic":
            graph_data = load_graph_data(graph_type)
            # removing edges already in the SL data
            SL_pos = SL_data[SL_data['label'] == True]
            SL_pos_list = sorted([tuple(r) for r in SL_pos[['gene1','gene2']].to_numpy()] + [tuple(r) for r in SL_pos[['gene2','gene1']].to_numpy()])
            graph_list = [tuple(r) for r in graph_data[['gene1','gene2']].to_numpy()]
            left = list(set(graph_list) - set(SL_pos_list))
            graph_data = pd.DataFrame(left, columns=['gene1','gene2'])
            if cell_name != "synlethdb":
                graph_data = graph_data[(graph_data["gene1"].isin(kept_genes))&(graph_data["gene2"].isin(kept_genes))]
        else:
            graph_data = load_graph_data(graph_type)
            if cell_name != "synlethdb":
                graph_data = graph_data[(graph_data["gene1"].isin(kept_genes))&(graph_data["gene2"].isin(kept_genes))]

        graph_data_list.append(graph_data)

    # merge, filter and mapping
    SL_data, graph_data_list, gene_mapping = merge_and_mapping(SL_data, graph_data_list, SL_genes)
    
    # generate node features
    x = choose_node_attribute(attr, gene_mapping, cell_name, graph_data_list)
    
    # generate torch data
    data_x = torch.tensor(x, dtype=torch.float)
    
    data_edge_index_list = []
    for graph_data in graph_data_list:
        temp_edge_index = torch.tensor([graph_data['gene1'].values, graph_data['gene2'].values], dtype=torch.long)
        data_edge_index_list.append(temp_edge_index)
        
    data = Data(x=data_x, edge_index_list=data_edge_index_list)

    SL_data_train['gene1'] = SL_data_train['gene1'].map(gene_mapping)
    SL_data_train['gene2'] = SL_data_train['gene2'].map(gene_mapping)
    SL_data_val['gene1'] = SL_data_val['gene1'].map(gene_mapping)
    SL_data_val['gene2'] = SL_data_val['gene2'].map(gene_mapping)
    SL_data_test['gene1'] = SL_data_test['gene1'].map(gene_mapping)
    SL_data_test['gene2'] = SL_data_test['gene2'].map(gene_mapping)
    
    return data, SL_data_train, SL_data_val, SL_data_test, gene_mapping