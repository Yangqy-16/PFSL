"""Build a reusable physical-PPI bank from frozen embeddings; never reads SL labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from coach_pl.configuration import CfgNode
from pfsl.ppi import physical_edges

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_bank(cfg, biogrid):
    seq_root, text_root = Path(cfg.DATASET.ESM_ROOT), Path(cfg.DATASET.TEXT_EMB_ROOT)
    if not seq_root.is_dir() or not text_root.is_dir():
        raise FileNotFoundError('Both frozen embedding directories must exist')
    with open(cfg.DATASET.TEXT_PATH) as stream:
        descriptions = json.load(stream)
    ids = sorted({int(p.stem) for root in (seq_root, text_root)
                  for p in root.glob('*.pt') if p.stem.isdigit()})
    if not ids:
        raise ValueError('No frozen embeddings found')
    aspects = (list(range(cfg.DATASET.ASPECT_NUM)) if cfg.MODEL.FUNC_USE_ASPECT is None
               else [int(i) for i in str(cfg.MODEL.FUNC_USE_ASPECT).split(',')])
    seq = torch.zeros(len(ids), cfg.MODEL.SEQ_IN_CHANNELS)
    text = torch.zeros(len(ids), cfg.MODEL.FUNC_IN_CHANNELS)
    available = torch.zeros(len(ids), 2)
    ignored_orphan_text_ids = []
    for i, gid in enumerate(ids):
        seq_file, text_file = seq_root / f'{gid}.pt', text_root / f'{gid}.pt'
        if seq_file.exists():
            value = torch.load(seq_file, map_location='cpu', weights_only=True).float()
            if value.ndim != 2 or value.shape[1] != seq.shape[1] or not len(value) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid residue sequence embedding: {seq_file}')
            seq[i] = value.mean(dim=0)
            available[i, 0] = 1
        desc = descriptions.get(str(gid))
        if text_file.exists() and desc is None:
            ignored_orphan_text_ids.append(gid)
        if text_file.exists() and desc is not None:
            value = torch.load(text_file, map_location='cpu', weights_only=True).float()
            if value.shape != (cfg.DATASET.ASPECT_NUM, text.shape[1]):
                raise ValueError(f'Invalid aspect embedding shape: {text_file}')
            if len(desc) != cfg.DATASET.ASPECT_NUM:
                raise ValueError(f'Missing aspect descriptions for {gid}')
            valid = [j for j in aspects if desc[j].strip()]
            if valid:
                selected = value[valid]
                if not torch.isfinite(selected).all():
                    raise ValueError(f'Nonfinite observed text embedding: {text_file}')
                text[i] = selected.mean(dim=0)
                available[i, 1] = 1
        if (i + 1) % 1000 == 0:
            print(f'Pooled frozen features: {i + 1}/{len(ids)}', flush=True)
    pairs, stats = physical_edges(biogrid, ids)
    neighbors = [[] for _ in ids]
    for a, b in pairs:
        neighbors[a].append(b)
        neighbors[b].append(a)
    ptr, indices = [0], []
    for row in neighbors:
        indices.extend(sorted(row))
        ptr.append(len(indices))
    stats.update({'sequence_available': int(available[:, 0].sum()),
                  'text_available': int(available[:, 1].sum()),
                  'nonisolated_nodes': sum(bool(n) for n in neighbors)})
    metadata = {'biogrid_path': str(Path(biogrid).resolve()), 'biogrid_sha256': sha256(biogrid),
                'text_path': str(Path(cfg.DATASET.TEXT_PATH).resolve()),
                'text_sha256': sha256(cfg.DATASET.TEXT_PATH),
                'sequence_root': str(seq_root.resolve()), 'text_embedding_root': str(text_root.resolve()),
                'aspects': aspects, 'stats': stats,
                'ignored_orphan_text_ids': ignored_orphan_text_ids,
                'protocol': 'Physical background graph only; no SL files/labels read. All background nodes visible.'}
    return {'version': 1, 'gene_ids': torch.tensor(ids), 'seq': seq, 'text': text,
            'available': available, 'indptr': torch.tensor(ptr),
            'indices': torch.tensor(indices, dtype=torch.long), 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'pfsl/configuration/prosl.yaml'))
    parser.add_argument('--biogrid', default=str(ROOT.parent / 'MVGCNiSL/data/BIOGRID-9606.csv'))
    parser.add_argument('--output', default=str(ROOT / 'data/ppi/physical_v1.pt'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; use --force only to rebuild deliberately')
    torch.set_num_threads(2)
    bank = build_bank(CfgNode.load_yaml_with_base(args.config), args.biogrid)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    torch.save(bank, temp)
    temp.replace(output)
    manifest = dict(bank['metadata'], bank_sha256=sha256(output))
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
