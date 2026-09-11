"""
This script is from ESMC's repository (https://github.com/biohub/esm).

Note that long sequences may take lots of memory when inferring. Please set 'batch_size' suitable for your device.

After running this script, you can load the generated ESMC representations of proteins by: 
    np.load('###.npy', allow_pickle=True).item()
"""

from pathlib import Path
import os
import argparse
from tqdm import tqdm
import pickle
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login

# login with your Hugging Face credentials
login(token='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')

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--root", type=Path, default='/data/qingyuyang/SL/PFSL/data/pretrained_emb')
    return parser.parse_args()


def select_undone_pdbs(input_list: Path, output_dir: Path) -> list[tuple[str, str]]:
    all_pdbs = [i[0] for i in input_list]  # gene_id
    done_pdbs = [int(i[:-4]) for i in os.listdir(output_dir) if i.endswith('.pt')]
    undone_pdbs = list(set(all_pdbs).difference(set(done_pdbs)))  # undone PDB names
    return [pair for pair in input_list if pair[0] in undone_pdbs]


if __name__ == "__main__":
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # batch_size = args.batch_size

    dataset_root = args.root
    save_path = f'{dataset_root}/esmc'  # NOTE: can change
    os.makedirs(save_path, exist_ok=True)

    with open(f'/data/qingyuyang/SL/PFSL/data/SLKB/all_id_seq.pkl', 'rb') as f:  # NOTE: can change
        all_list = pickle.load(f)
    print('Finish loading protein data')

    all_list = select_undone_pdbs(all_list, save_path)

    # # example GFP sequence
    sequences = [('example', "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")]

    model = AutoModelForMaskedLM.from_pretrained(
        "/data/qingyuyang/SL/PFSL/pfsl_process/ESMC",
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

    for idx, seq in tqdm(sequences): #all_list
        inputs = tokenizer([seq], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        embedding = output.last_hidden_state[0]

        L = embedding.shape[0]
        embedding = embedding[1:L-1, :]
        print(embedding.shape)

        # torch.save(embedding, f'{save_path}/{idx}.pt')