import os
import json
import torch

FUNC_USE_ATT = True
TEXT_EMB_ROOT = '/data/qingyuyang/SL/PFSL/data/pretrained_emb/biolinkbert'
TEXT_PATH = '/data/qingyuyang/SL/PFSL/data/prot_func/all_text.txt'

with open(TEXT_PATH, 'r') as f:
    all_texts = json.load(f)


def text_emb(gid: int, aspect_total_num: int = 24):
    # unuse_aspects = [int(i) for i in str(self.cfg.MODEL.FUNC_UNUSE_ASPECT).split(',')] if self.cfg.MODEL.FUNC_UNUSE_ASPECT is not None else []
    use_aspects = sorted(list(set(range(aspect_total_num)))) #.difference(set(unuse_aspects))

    if os.path.exists(os.path.join(TEXT_EMB_ROOT, f"{gid}.pt")):
        g_func = torch.load(os.path.join(TEXT_EMB_ROOT, f"{gid}.pt"), map_location='cpu', weights_only=True)
        
        use_indices = torch.tensor(use_aspects, dtype=torch.long)
        g_func = g_func.index_select(0, use_indices)
    
        g_text = all_texts[str(gid)]
        g_text = [g_text[i] for i in use_aspects]

        g_mask = torch.zeros(len(use_aspects), dtype=torch.bool)

        for i in range(len(g_text)):
            if g_text[i] == '':
                g_mask[i] = True
    else:
        g_func = torch.randn((len(use_aspects), 1024))  # Random initialization
        g_mask = torch.ones(len(use_aspects), dtype=torch.bool)

    if FUNC_USE_ATT:
        return g_func, g_mask
    else:
        valid_mask = ~g_mask
        if valid_mask.any():
            g_func = g_func[valid_mask].mean(dim=0)
        else:
            g_func = g_func.mean(dim=0)
        return g_func, None