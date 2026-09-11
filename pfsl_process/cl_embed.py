# 简单的特征提取代码示例 (PyTorch)
import os
import torch
from transformers import AutoTokenizer, AutoModel

# if __name__ == '__main__':
OUTPUT_DIR = './cl_embed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 推荐使用专门在 PubMed 上微调过的模型
model_name = "/data/qingyuyang/SL/PFSL/pfsl_process/PubmedBERT" #"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

def get_text_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = encoder(**inputs)
    # 提取 [CLS] token 作为整个细胞系的句向量表征
    return outputs.last_hidden_state[:, 0, :] 

with open('./llm_text_new.txt', 'r') as f:
    cell_lines = f.readlines()

for cl in cell_lines:
    embedding = get_text_embedding(cl)
    cl_name = cl.split()[0]
    torch.save(embedding, f'{OUTPUT_DIR}/{cl_name}.pt')