from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import os
import json
import copy
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch.optim import lr_scheduler

from model import GCN_pool, GateModule, PF_Att
from omics_data import generate_feature
from text_data import *
from metric import compute_metrics

ASPECT_NUM = 24


def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=1)

    parser.add_argument('--specific_graph', type=lambda s:[item for item in s.split("%") if item != ""], default=["SL"], help="lists of cell-specific graphs to use.")
    parser.add_argument(
        "--indep_graph",
        type=lambda s: [item for item in s.split("%") if item != ""],
        default=["PPI-physical", "co-exp", "co-ess"], #"PPI-genetic", 
        help="lists of cell-independent graphs to use.",
    )
    parser.add_argument('--node_feats', type=str, default="raw_omics", help="gene node features")

    # parser.add_argument('--text_emb_root', type=Path)

    # parser.add_argument('--model', type=str, default="GCN_pool", help="model type")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="number of maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=10, help="patience in early stopping")

    parser.add_argument('--input_dir', type=Path, default='/data/qingyuyang/SL/MVGCNiSL/sl_data/specific')
    parser.add_argument('--output_dir', type=Path, default='./output')
    parser.add_argument('--cell_line', type=str, default="A549", help="which cell line to train and predict")
    parser.add_argument('--scene', type=str, default="C1")
    parser.add_argument('--fold', type=str, default=0)
    # parser.add_argument('--target_cl', type=str, default=None)

    args = parser.parse_args()
    return args


def process_epoch(loader, gnn_model, gate_module, pf_model, device, num_graphs, text_embeddings, text_masks, args, optimizer=None):
    """
    统一的训练/评估函数
    如果传入了 optimizer，则视为训练阶段 (Train)
    如果 optimizer=None，则视为评估阶段 (Val/Test)
    """
    # 1. 根据是否有 optimizer 来判断是否为训练模式
    is_train = optimizer is not None
    
    # 2. 设置模型状态
    if is_train:
        gnn_model.train()
        gate_module.train()
        pf_model.train()
        desc = "Train"
    else:
        gnn_model.eval()
        gate_module.eval()
        pf_model.eval()
        desc = "Evaluate"

    total_loss = 0
    total_samples = 0

    all_logits, all_labels = [], []
    g1_indices, g2_indices = [], []
    
    # 3. 动态控制梯度上下文！如果是 train，相当于什么都不加；如果是 eval，相当于 with torch.no_grad()
    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=desc)
        for batch in bar:
            batch = batch.to(device)
            x = batch.x

            if is_train:
                optimizer.zero_grad()

            # ================= 前向传播 (步骤 A, B, C 完全复用) =================
            temp_z_list = []
            for i in range(num_graphs):
                mask = (batch.edge_type == i)
                sub_edge_index = batch.edge_index[:, mask]
                temp_z = gnn_model.encode(x, sub_edge_index)
                temp_z_list.append(temp_z)

            z_stacked = torch.stack(temp_z_list, dim=1)
            if args.pooling == "max":
                z = z_stacked.max(dim=1)[0]
            elif args.pooling == "mean":
                z = z_stacked.mean(dim=1)
            else:
                raise ValueError("Invalid pooling method!")

            src_idx = batch.edge_label_index[0] 
            dst_idx = batch.edge_label_index[1]
            # print(src_idx, dst_idx)
            z_src = z[src_idx]
            z_dst = z[dst_idx]

            gate_weights = gate_module(z_src, z_dst).unsqueeze(-1)  # [bs, num_aspects, 1]
            
            real_src_id = batch.n_id[src_idx]
            real_dst_id = batch.n_id[dst_idx]

            text_src, text_src_mask = text_embeddings[real_src_id], text_masks[real_src_id]  # [bs, num_aspects, 1024], [bs, num_aspects]
            text_src, text_src_mask = text_src.to(device), text_src_mask.to(device)
            text_dst, text_dst_mask = text_embeddings[real_dst_id], text_masks[real_dst_id]  # [bs, num_aspects, 1024], [bs, num_aspects]
            text_dst, text_dst_mask = text_dst.to(device), text_dst_mask.to(device)

            gated_text_src = text_src# * gate_weights  # [bs, num_aspects, 1024]
            gated_text_dst = text_dst# * gate_weights  # [bs, num_aspects, 1024]

            logits, atts = pf_model(gated_text_src, text_src_mask, gated_text_dst, text_dst_mask)
            
            # 计算 Loss
            labels = batch.edge_label.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # ================= 仅在训练时反向传播 =================
            if is_train:
                loss.backward()
                optimizer.step()

            # ================= 记录指标 =================
            batch_size = batch.edge_label.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.append(logits.clone().detach().cpu())
            all_labels.append(labels.clone().detach().cpu())
            g1_indices.append(real_src_id)
            g2_indices.append(real_dst_id)
            
            # 动态更新进度条
            bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_logits = torch.cat(all_logits)
    all_logits = torch.sigmoid(all_logits)  # for cls !
    all_labels = torch.cat(all_labels)
    g1_indices = torch.cat(g1_indices)
    g2_indices = torch.cat(g2_indices)
    return total_loss / total_samples, all_logits, all_labels, g1_indices, g2_indices


if __name__ == "__main__":
    print('Start!')
    args = init_argparse()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    output_path = f"{args.output_dir}/{args.cell_line}/{args.scene}"
    os.makedirs(output_path, exist_ok=True)

    data, SL_data_train, SL_data_val, SL_data_test, gene_mapping = \
        generate_feature(args.cell_line, f"{args.input_dir}/{args.cell_line}/{args.scene}", args.fold)
    
    # 假设您已加载了由 LLM 离线提取的文本表征字典或 Tensor
    # text_embeddings: 形状为 [全量基因数, 24, 1024]
    with open('/data/qingyuyang/SL/PFSL/data/mapping/name2id.pkl', 'rb') as f:
        gn2gid = pickle.load(f)  # maps gene name to gene id
    
    uid2textemb = torch.zeros((len(gene_mapping), 24, 1024), dtype=torch.float32, device=device)
    uid2textmask = torch.zeros((len(gene_mapping), 24), dtype=bool, device=device)
    
    for gn, uid in tqdm(gene_mapping.items()):
        emb, mask = text_emb(gn2gid[gn], ASPECT_NUM)

        uid2textemb[uid] = emb
        uid2textmask[uid] = mask

    # ======= 新增代码：将 edge_index_list 合并为 edge_index 和 edge_type =======
    all_edges = []
    all_types = []
    num_graphs = len(data.edge_index_list)

    for i, ei in enumerate(data.edge_index_list):
        all_edges.append(ei)
        # 为当前图的每条边打上对应的类别标签 i
        all_types.append(torch.full((ei.size(1),), i, dtype=torch.long))

    # 覆盖 data 中默认的 edge_index，并添加 edge_type
    data.edge_index = torch.cat(all_edges, dim=1)
    data.edge_type = torch.cat(all_types, dim=0)

    # 删除 edge_index_list 避免后续误用
    del data.edge_index_list

    # 2. 准备您当前 batch 需要预测的 SL 数据 (训练集)
    sl_train_edge_index = torch.tensor(SL_data_train[['gene1', 'gene2']].to_numpy(dtype=int).T)
    sl_train_labels = torch.tensor(SL_data_train['label'], dtype=torch.float32)
    sl_val_edge_index = torch.tensor(SL_data_val[['gene1', 'gene2']].to_numpy(dtype=int).T)
    sl_val_labels = torch.tensor(SL_data_val['label'], dtype=torch.float32)
    sl_test_edge_index = torch.tensor(SL_data_test[['gene1', 'gene2']].to_numpy(dtype=int).T)
    sl_test_labels = torch.tensor(SL_data_test['label'], dtype=torch.float32)

    # 3. 初始化 LinkNeighborLoader
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[15, 10], # 假设您的 GCN 有 2 层。第一层采样 15 个邻居，第二层采样 10 个。
        edge_label_index=sl_train_edge_index, # Loader 将根据这些边来采样邻居
        edge_label=sl_train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[15, 10], # 假设您的 GCN 有 2 层。第一层采样 15 个邻居，第二层采样 10 个。
        edge_label_index=sl_val_edge_index, # Loader 将根据这些边来采样邻居
        edge_label=sl_val_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[15, 10], # 假设您的 GCN 有 2 层。第一层采样 15 个邻居，第二层采样 10 个。
        edge_label_index=sl_test_edge_index, # Loader 将根据这些边来采样邻居
        edge_label=sl_test_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 初始化模块
    gnn_model = GCN_pool(in_channels=data.x.shape[1], out_channels=args.out_channels).to(device)
    gate_module = GateModule(gnn_dim=args.out_channels).to(device)
    pf_model = PF_Att().to(device) # 您现有的基于文本的模型

    optimizer = torch.optim.Adam(
        list(pf_model.parameters()), #list(gnn_model.parameters()) + list(gate_module.parameters()) + 
        lr=args.LR
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.patience
    best_epoch = -1

    best_gnn_state, best_gate_state, best_pf_state = None, None, None

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")
        
        # 一行代码搞定训练！(传入了 optimizer)
        train_loss, _, _, _, _ = process_epoch(train_loader, gnn_model, gate_module, pf_model, device, num_graphs, uid2textemb, uid2textmask, args, optimizer=optimizer)
        
        # 一行代码搞定验证！(没有传入 optimizer，默认为 None)
        val_loss, _, _, _, _ = process_epoch(val_loader, gnn_model, gate_module, pf_model, device, num_graphs, uid2textemb, uid2textmask, args)
        
        scheduler.step(val_loss)

        print(f"[Result] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping 逻辑 (保持不变)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_gnn_state = copy.deepcopy(gnn_model.state_dict())
            best_gate_state = copy.deepcopy(gate_module.state_dict())
            best_pf_state = copy.deepcopy(pf_model.state_dict())
            print(f" -> ⭐ Best model updated!")
        else:
            patience_counter += 1
            print(f" -> Early stopping counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print(f"=====================================")
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}!")
                break

    # torch.save(best_gnn_state, f'{output_path}/gnn.pt')
    # torch.save(best_gate_state, f'{output_path}/gate.pt')
    # torch.save(best_pf_state, f'{output_path}/pf.pt')

    # 测试阶段
    print("\n" + "="*40)
    print(f"Loading best model from Epoch {best_epoch+1}...")
    gnn_model.load_state_dict(best_gnn_state)
    gate_module.load_state_dict(best_gate_state)
    pf_model.load_state_dict(best_pf_state)

    # 一行代码搞定测试！
    test_loss, test_logits, test_labels, test_g1, test_g2 = process_epoch(test_loader, gnn_model, gate_module, pf_model, device, num_graphs, uid2textemb, uid2textmask, args)
    
    test_metrics = compute_metrics(test_labels, test_logits)
    with open(f'{output_path}/{args.fold}.json', "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    
    df = pd.DataFrame({'gene1': test_g1.detach().cpu().numpy(), 'gene2': test_g2.detach().cpu().numpy(), 'prob': test_logits.detach().cpu().numpy()})
    gene_reverse_mapping = {uid: gn for gn, uid in gene_mapping.items()}
    df['gene1'] = df['gene1'].map(gene_reverse_mapping)
    df['gene2'] = df['gene2'].map(gene_reverse_mapping)
    df.to_csv(f'{output_path}/{args.fold}.csv', index=False)

    print(f"🚀 Final Test Loss: {test_loss:.4f}")
    print("="*40)