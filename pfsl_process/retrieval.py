import json
import pickle
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch
import faiss
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProteinRetriever:
    def __init__(self, embedding_dim=1024):
        self.dim = embedding_dim
        self.gene_ids = []  # 记录 index -> gene_name
        self.index = None
        
    def preprocess_tensors(self, gene_data_dict):
        """
        处理原始数据：Filter None -> Stack -> Mean Pooling -> L2 Norm
        """
        vectors = []
        valid_gene_ids = [] # 使用临时列表，确保只有有效向量的基因才被加入
        
        print("正在处理蛋白质向量 (Filtering & Pooling)...")
        
        for gene, tensor_list in gene_data_dict.items():
            if not tensor_list:
                continue
            
            # --- 修改点 1: 过滤 None 值 ---
            # 同时也确保 tensor 维度正确 (如果是 [1, 1024] 变成 [1024])
            valid_tensors = []
            for t in tensor_list:
                if t is not None:
                    # 统一维度: 如果是 [1, 1024] -> squeeze 成 [1024]
                    # 如果是 [1024] -> 保持不变
                    t_sq = t.squeeze() 
                    if t_sq.dim() == 0: # 极其罕见的标量情况，跳过
                         continue
                    valid_tensors.append(t_sq)
            
            # --- 修改点 2: 边界情况处理 (全为 None) ---
            if len(valid_tensors) == 0:
                # print(f"警告: 基因 {gene} 没有有效的功能向量，已跳过。")
                continue

            # 3. Stack & Mean Pooling
            # stack 后维度: [N_valid_aspects, 1024]
            stacked = torch.stack(valid_tensors)
            
            # 如果只有一个 aspect，mean(dim=0) 会导致维度降为 Scalar吗？不会。
            # [1, 1024] -> mean(0) -> [1024] (正确)
            # [5, 1024] -> mean(0) -> [1024] (正确)
            if stacked.dim() == 1: 
                mean_vec = stacked # 只有一个向量的情况
            else:
                mean_vec = torch.mean(stacked, dim=0)
            
            # 转为 numpy
            vec_np = mean_vec.detach().cpu().numpy()
            
            # 4. L2 Normalization
            norm = np.linalg.norm(vec_np) + 1e-10
            vec_normalized = vec_np / norm
            
            vectors.append(vec_normalized)
            valid_gene_ids.append(gene)
            
        # 更新类属性
        self.gene_ids = valid_gene_ids
        
        if len(vectors) == 0:
            raise ValueError("没有任何基因包含有效向量，无法构建索引！")

        # 转换为 FAISS 需要的 float32 矩阵 [Num_Genes, 1024]
        matrix = np.array(vectors).astype('float32')
        return matrix

    def build_index(self, gene_data_dict):
        """构建 FAISS 索引"""
        matrix = self.preprocess_tensors(gene_data_dict)
        num_vectors = matrix.shape[0]
        
        print(f"构建索引中，共 {num_vectors} 个有效基因...")
        
        # Inner Product (在归一化向量上等价于 Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)
        print("索引构建完成！")

    def search(self, query_gene_tensor_list, top_k=1000):
        """检索函数"""
        if self.index is None:
            raise ValueError("请先调用 build_index 构建索引！")
            
        # --- 修改点 3: Query 处理逻辑同上 ---
        valid_tensors = []
        if query_gene_tensor_list:
            for t in query_gene_tensor_list:
                if t is not None:
                    valid_tensors.append(t.squeeze())
        
        if len(valid_tensors) == 0:
            print("警告: Query 基因没有任何有效向量。")
            return []
            
        stacked = torch.stack(valid_tensors)
        if stacked.dim() == 1:
            mean_vec = stacked
        else:
            mean_vec = torch.mean(stacked, dim=0)
            
        vec_np = mean_vec.detach().cpu().numpy()
        norm = np.linalg.norm(vec_np) + 1e-10
        
        # reshape 成 [1, 1024] 供 FAISS 查询
        query_vec = (vec_np / norm).reshape(1, -1).astype('float32')
        
        D, I = self.index.search(query_vec, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1: continue 
            # 注意：这里的 idx 是 matrix 的行号，对应 self.gene_ids
            gene_name = self.gene_ids[idx]
            results.append((gene_name, float(score)))
            
        return results


# 假设这 22 个 Aspect 是固定的，顺序如下 (必须与你生成的 Tensor list 顺序严格对应)
ASPECT_NAMES = [
    "Activity regulation", "Developmental stage", "Induction", "GO (bio process)", 
    "GO (cell comp)", "GO (mol func)", "Involvement in disease", "Subcellular location", 
    "Domain", "Pathway", "Cofactor", "Catalytic activity", "EC number", "Caution", 
    "Function", "Miscellaneous", "Polymorphism", "PTM", "Proteomes", 
    "Sequence similarities", "Subunit structure", "Tissue specificity"
]

class WeightedMaxSimReranker:
    def __init__(self, aspect_names=ASPECT_NAMES, device=DEVICE):
        self.aspect_names = aspect_names
        self.num_aspects = len(aspect_names)
        self.device = device
        
        # 初始化权重为 1.0 (全等)
        self.aspect_weights = torch.ones(self.num_aspects).to(self.device)
        self.idf_computed = False

    def compute_idf_weights(self, idf_weights, manual_boost=None): #gene_data_dict
        """
        第一步：扫描整个数据库，计算 IDF 权重。
        
        Args:
            gene_data_dict: dict, { 'GeneName': [tensor_aspect_0, None, tensor_aspect_2, ...] }
                            注意：列表中如果某个 Aspect 缺失，建议设为 None 或 全0向量。
            manual_boost: dict, { 'AspectName': 2.0 } 
                          允许人工给某些关键 Aspect (如 Pathway) 强行加权。
        """
        # print("正在计算 TF-IDF 权重...")
        # N_total = len(gene_data_dict)
        # doc_freq = torch.zeros(self.num_aspects)

        # # 1. 统计文档频率 (Document Frequency)
        # for gene, tensor_list in gene_data_dict.items():
        #     for i, tensor in enumerate(tensor_list):
        #         # 判断该 Aspect 是否存在 (不为 None 且不全为 0)
        #         if tensor is not None:
        #             if isinstance(tensor, torch.Tensor) and torch.count_nonzero(tensor) > 0:
        #                 doc_freq[i] += 1
        
        # # 2. 计算 IDF: log(N / (df + 1))
        # # 加上 1.0 基础分，保证即使所有蛋白都有该 Aspect，权重也不会变成 0
        # idf_weights = torch.log(N_total / (doc_freq + 1)) + 1.0
        
        # 3. (可选) 人工干预权重
        if manual_boost:
            for name, boost_val in manual_boost.items():
                if name in self.aspect_names:
                    idx = self.aspect_names.index(name)
                    print(f"  -> 人工增强 {name}: 原始IDF={idf_weights[idx]:.2f}, 增强后={idf_weights[idx] * boost_val:.2f}")
                    idf_weights[idx] *= boost_val
        
        # 归一化权重 (可选，让总和在一个量级，方便调试)
        # idf_weights = idf_weights / idf_weights.mean()
        
        self.aspect_weights = idf_weights.to(self.device)
        self.idf_computed = True
        print("权重计算完成！\nTop 3 重要 Aspects:", 
              [self.aspect_names[i] for i in torch.topk(self.aspect_weights, 3).indices])

    def rerank(self, query_tensors, candidate_batch_dict):
        """
        第二步：精排。计算 Query 与一批 Candidates 的 Weighted MaxSim 分数。
        
        Args:
            query_tensors: list of 22 tensors (Query Gene)
            candidate_batch_dict: dict of { 'GeneName': list of 22 tensors } (候选集，比如 Top-50)
        
        Returns:
            sorted_results: list of (GeneName, Score, Match_Details)
        """
        if not self.idf_computed:
            print("警告: 尚未计算 IDF，将使用默认全 1 权重。")

        # --- 1. 准备 Query Tensor [1, 22, 1024] ---
        # 缺失的 aspect 补零，以免计算出错
        q_stack = self._stack_tensors(query_tensors).unsqueeze(0).to(self.device) 
        
        # --- 2. 准备 Candidate Tensor Batch [B, 22, 1024] ---
        cand_names = list(candidate_batch_dict.keys())
        cand_stack_list = [self._stack_tensors(candidate_batch_dict[name]) for name in cand_names]
        c_batch = torch.stack(cand_stack_list).to(self.device)
        
        # --- 3. 计算 MaxSim ---
        
        # (a) 归一化 (Cosine Similarity 需要)
        q_norm = F.normalize(q_stack, p=2, dim=2) # [1, 22, 1024]
        c_norm = F.normalize(c_batch, p=2, dim=2) # [B, 22, 1024]
        
        # (b) 计算交互矩阵: Query 的每个 Aspect 对 Candidate 的每个 Aspect
        # [1, 22, 1024] @ [B, 1024, 22] -> [B, 22, 22]
        # sim_matrix[b, i, j] 表示 Query Aspect i 和 Candidate Aspect j 的相似度
        sim_matrix = torch.matmul(q_norm, c_norm.transpose(1, 2)) # Broadcast机制会自动处理 batch
        
        # (c) Max 操作: 对于 Query 的每个 Aspect i，找 Candidate 中最匹配的 j
        # max_scores: [B, 22]
        max_scores, _ = torch.max(sim_matrix, dim=2)
        
        # (d) 加权求和: 乘以 Query 侧的 IDF 权重
        # weights: [22] -> [1, 22]
        weights = self.aspect_weights.unsqueeze(0)
        
        # 这里的 mask 是为了处理 Query 本身缺失 Aspect 的情况 (全0向量相似度由 max 算出可能是 0，但为了保险)
        # 如果 Query 的某个 tensor 是全 0，它算出的 max_score 也是 0，这步其实是隐式包含的
        weighted_scores = max_scores * weights
        
        # (e) 最终得分: [B]
        final_scores = torch.sum(weighted_scores, dim=1)
        
        # --- 4. 格式化输出 ---
        results = []
        for idx, score in enumerate(final_scores):
            results.append((cand_names[idx], score.item()))
            
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _stack_tensors(self, tensor_list):
        """辅助函数：将 list 转为 [22, 1024] tensor，处理 None"""
        processed = []
        for t in tensor_list:
            if t is None:
                processed.append(torch.zeros(1024, device=self.device))
            else:
                processed.append(t if t.dim()==1 else t.squeeze())
        return torch.stack(processed)


# --- 模拟使用示例 ---
if __name__ == "__main__":
    ##### --- Preprocess: build gene data dict --- #####
    print('Loading data ...')

    cl = '22RV1'
    ga = '7048'
    query_g = '5599'

    scene = pd.read_csv(f'/data/qingyuyang/ESM4SL/data/SLKB/case/{cl}/{ga}/sl_test_0.csv')
    pos_num = len(scene[scene['2'] == 1])  # TP 数量
    print(f"查询基因 {ga} 的未知 SL 伙伴数量: {pos_num - 1}")
    
    gt_dict = {}
    for _, contents in scene.iterrows():
        gt_dict[str(contents['0'])] = contents['2']
        gt_dict[str(contents['1'])] = contents['2']
    
    genes = list(set([str(i) for i in scene['0']]) | set([str(i) for i in scene['1']]))
    genes.remove(ga)
    # print(genes)
    
    # NOTE: run after 1st time
    with open('./retrieval/all_gene_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # print(type(data))
    data = {key: data[key] for key in genes}  # NOTE: 只保留检索库里存在的基因数据，避免后续检索时 KeyError

    ################### NOTE: 1st run ###################
    # gene_txt = json.load(open('/data/qingyuyang/ESM4SL/data/prot_func/all_text.txt', 'r'))

    # data = {}
    # for gid, txt in tqdm(gene_txt.items(), total=len(gene_txt)):
    #     assert type(txt) == list
    #     embedding = torch.load(f'/data/qingyuyang/ESM4SL/data/protrek_text_emb/{gid}.pt', weights_only=True, map_location=DEVICE)
    #     assert embedding.shape == (22, 1024)

    #     output = []
    #     for i, aspect in enumerate(txt):
    #         if aspect != '':
    #             output.append(embedding[i])
    #             doc_freq[i] += 1
    #         else:
    #             output.append(None)
        
    #     assert len(output) > 0
    #     data[gid] = output
    
    # with open('./retrieval/all_gene_data.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    ########################################################
    
    print('Computing IDF weights ...')
    N_total = len(data)
    doc_freq = torch.zeros(len(ASPECT_NAMES))

    for gid, contents in data.items():
        for i, content in enumerate(contents):
            if content is not None and torch.count_nonzero(content) > 0:
                doc_freq[i] += 1

    idf_weights = torch.log(N_total / (doc_freq + 1)) + 1.0
    # torch.save(idf_weights, './retrieval/idf_weights.pt')
    # idf_weights = torch.load('./retrieval/idf_weights.pt', weights_only=True)
    idf_weights = torch.zeros_like(idf_weights) + 1.0  # 如果不想用 IDF 权重，直接全设为 1.0
    # ----------------------------------------

    with open('/data/qingyuyang/ESM4SL/data/mapping/name2id.pkl', 'rb') as f:
        gn2gid = pickle.load(f)
    gid2gn = {v: k for k, v in gn2gid.items()}

    # 1. 实例化
    retriever = ProteinRetriever()
    
    # 2. 构建索引 (这一步通常只做一次，可以把 index 保存到硬盘)
    retriever.build_index(data)
    
    # 3. 检索
    # 拿 TP53 去搜库里的相似蛋白
    query_data = data[query_g]

    # --- Step 1: 粗筛 (Coarse Retrieval) ---
    # 这一步极快，把几万个基因变成 100 个
    K = 100
    top_results = retriever.search(query_data, top_k=K)
    
    # print(f"\n检索结果 (Top {K}):")
    # for gene, score in top_results:
    #     print(f"Gene: {gid2gn[gene]}, Cosine Similarity: {score:.4f}")

    # 准备精排数据
    candidates_dict = { name: data[name] for name, _ in top_results }

    # --- Step 2: 精排 (Fine-grained Reranking) ---
    # 这一步使用 GPU 进行矩阵运算，计算 MaxSim
    # 假设 all_genes_data 是你整理好的所有数据
    reranker = WeightedMaxSimReranker()

    # 这一步体现了 TF-IDF 思想
    # 同时，我们利用生物学先验知识，给 Pathway 和 Disease 强行加权 2.0 倍
    reranker.compute_idf_weights(idf_weights, manual_boost={
        "Pathway": 1.5, 
        "Involvement in disease": 1.5,
        "Function": 1.5
    })

    reranked_results = reranker.rerank(query_data, candidates_dict)

    # --- Step 3: 打印结果 ---
    k = 10
    recall_k = -1
    
    print("最终推荐 SL 伙伴:")
    for gene, score in reranked_results[:k]:
        label = gt_dict[gene]
        if label == 1:
            recall_k += 1
        # print(f"Gene: {gene}, Weighted Score: {score:.4f}, GT: {label}")  #gid2gn[]
    
    print(f"Recall@{k}: {recall_k / pos_num:.4f}")
