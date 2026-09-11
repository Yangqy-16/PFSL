import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda"
EMBED_DIM = 1024

tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large').eval().to(device)

with open('/data/qingyuyang/SL/ESM4SL/data/prot_func/all_text.txt', 'r') as f:
    pro_texts = json.load(f)

out_path = '/data/qingyuyang/SL/ESM4SL/data/pretrained_emb/biolinkbert'
os.makedirs(out_path, exist_ok=True)

# exist_gids = [i.split('.')[0] for i in os.listdir('/data/qingyuyang/SL/ESM4SL/data/pretrained_emb/biolinkbert')]

with torch.no_grad():
    for gid, texts in tqdm(pro_texts.items()):
        # if str(gid) in exist_gids:
        #     continue
        out_tensor = []

        for t in texts:
            if t:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(device)
                outputs = model(**inputs, output_hidden_states=True)
                out_tensor.append(outputs.hidden_states[-1][0].mean(dim=0).clone().detach().cpu())
            else:
                out_tensor.append(torch.zeros(EMBED_DIM, device='cpu', dtype=torch.float))

        torch.save(torch.stack(out_tensor), f'{out_path}/{gid}.pt')
        # break