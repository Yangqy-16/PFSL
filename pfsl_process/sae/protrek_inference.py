"""Inference-only wrapper around the exact ProTrek text representation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from transformers import BertTokenizerFast


DEFAULT_PROTREK_REPO = Path("/data/qingyuyang/prot_design/ProTrek")
DEFAULT_PROTREK_WEIGHTS = DEFAULT_PROTREK_REPO / "weights/ProTrek_650M"
REPRESENTATION_DIM = 1024
NORM_TOLERANCE = 1e-5


class TextRepresentationModel(Protocol):
    def get_text_repr(self, texts: list[str]) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class ProTrekPaths:
    repo_root: Path
    weights_root: Path
    protein_config: Path
    text_config: Path
    structure_config: Path
    checkpoint: Path


@dataclass(frozen=True)
class TokenizedText:
    """Exact fast-tokenizer view used only to construct text perturbations."""

    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_token_id: int

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be a built-in str")
        length = len(self.token_ids)
        if not (
            length
            and len(self.tokens) == length
            and len(self.offsets) == length
            and len(self.special_tokens_mask) == length
            and len(self.attention_mask) == length
        ):
            raise ValueError("tokenized fields must have the same nonzero length")
        if type(self.mask_token_id) is not int or self.mask_token_id < 0:
            raise ValueError("mask_token_id must be a nonnegative built-in int")
        previous_end = 0
        for index, (token_id, offset, special, attended) in enumerate(
            zip(
                self.token_ids,
                self.offsets,
                self.special_tokens_mask,
                self.attention_mask,
            )
        ):
            if type(token_id) is not int or token_id < 0:
                raise ValueError("token IDs must be nonnegative built-in ints")
            if (
                not isinstance(offset, tuple)
                or len(offset) != 2
                or any(type(value) is not int for value in offset)
            ):
                raise ValueError("offsets must be integer pairs")
            start, end = offset
            if start < 0 or end < start or end > len(self.text):
                raise ValueError("token offset lies outside the original text")
            if special not in (0, 1) or type(special) is not int:
                raise ValueError("special_tokens_mask must contain only built-in 0/1")
            if type(attended) is not int or attended != 1:
                raise ValueError("TokenizedText contains a structurally padded token")
            if special:
                if offset != (0, 0):
                    raise ValueError("special token offset must be exactly (0, 0)")
            else:
                if end <= start:
                    raise ValueError(f"non-special token {index} has no source substring")
                if start < previous_end:
                    raise ValueError("non-special offsets must be ordered and nonoverlapping")
                previous_end = end


def resolve_protrek_paths(
    repo_root: Path = DEFAULT_PROTREK_REPO,
    weights_root: Path = DEFAULT_PROTREK_WEIGHTS,
) -> ProTrekPaths:
    repo_root = Path(repo_root).resolve()
    weights_root = Path(weights_root).resolve()
    paths = ProTrekPaths(
        repo_root=repo_root,
        weights_root=weights_root,
        protein_config=weights_root / "esm2_t33_650M_UR50D",
        text_config=(
            weights_root / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ),
        structure_config=weights_root / "foldseek_t30_150M",
        checkpoint=weights_root / "ProTrek_650M.pt",
    )
    required = {
        "ProTrek repository": paths.repo_root,
        "ProTrek model module": (
            paths.repo_root / "model/ProTrek/protrek_trimodal_model.py"
        ),
        "protein config": paths.protein_config,
        "text config": paths.text_config,
        "structure config": paths.structure_config,
        "checkpoint": paths.checkpoint,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing ProTrek {label}: {path}")
    for label, path in (
        ("protein config", paths.protein_config),
        ("text config", paths.text_config),
        ("structure config", paths.structure_config),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"ProTrek {label} is not a directory: {path}")
    if not paths.checkpoint.is_file():
        raise FileNotFoundError(f"ProTrek checkpoint is not a file: {paths.checkpoint}")
    return paths


def load_protrek_model(paths: ProTrekPaths, device: str) -> torch.nn.Module:
    """Load ProTrek exactly as sl.py while disabling unused training metrics."""
    repo = str(paths.repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

    class ProTrekInferenceModel(ProTrekTrimodalModel):
        def initialize_metrics(self, stage: str) -> dict:
            return {}

    model = ProTrekInferenceModel(
        protein_config=str(paths.protein_config),
        text_config=str(paths.text_config),
        structure_config=str(paths.structure_config),
        from_checkpoint=str(paths.checkpoint),
    )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


class ProTrekTextEmbedder:
    """Batch text encoding with strict representation validation."""

    def __init__(
        self,
        model: TextRepresentationModel,
        device: str,
        paths: ProTrekPaths | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self._offset_tokenizer: BertTokenizerFast | None = None
        if paths is not None and (paths.text_config / "vocab.txt").is_file():
            try:
                tokenizer = BertTokenizerFast.from_pretrained(
                    str(paths.text_config), local_files_only=True
                )
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    "Cannot load BertTokenizerFast from the frozen ProTrek text config"
                ) from error
            if tokenizer.mask_token_id is None:
                raise ValueError("Frozen ProTrek tokenizer has no [MASK] token ID")
            self._offset_tokenizer = tokenizer
        self.provenance: dict[str, object] = {
            "protrek": {
                "checkpoint": str(paths.checkpoint) if paths else "unknown",
                "protein_config": str(paths.protein_config) if paths else "unknown",
                "text_config": str(paths.text_config) if paths else "unknown",
                "structure_config": (
                    str(paths.structure_config) if paths else "unknown"
                ),
            },
            "representation": {
                "method": "ProTrekTrimodalModel.get_text_repr",
                "dimension": REPRESENTATION_DIM,
            },
            "environment": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_prefix": str(Path(sys.prefix).resolve()),
                "python_version": sys.version.split()[0],
                "torch_version": str(torch.__version__),
            },
            "device": device,
        }

    @staticmethod
    def _active_ids(encoded: object) -> tuple[int, ...]:
        if not isinstance(encoded, dict) and not hasattr(encoded, "get"):
            raise ValueError("tokenizer returned an invalid encoding")
        raw_ids = encoded.get("input_ids")
        raw_attention = encoded.get("attention_mask")
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.detach().cpu().tolist()
        if isinstance(raw_attention, torch.Tensor):
            raw_attention = raw_attention.detach().cpu().tolist()
        if isinstance(raw_ids, list) and len(raw_ids) == 1 and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]
        if (
            isinstance(raw_attention, list)
            and len(raw_attention) == 1
            and isinstance(raw_attention[0], list)
        ):
            raw_attention = raw_attention[0]
        if (
            not isinstance(raw_ids, list)
            or not isinstance(raw_attention, list)
            or len(raw_ids) != len(raw_attention)
            or any(type(value) is not int for value in raw_ids)
            or any(type(value) is not int or value not in (0, 1) for value in raw_attention)
        ):
            raise ValueError("tokenizer returned invalid IDs/attention mask")
        seen_padding = False
        active: list[int] = []
        for token_id, attention in zip(raw_ids, raw_attention):
            if attention == 0:
                seen_padding = True
            elif seen_padding:
                raise ValueError("tokenizer attention mask reactivates after padding")
            else:
                active.append(token_id)
        return tuple(active)

    @classmethod
    def _ids_from_model_tokenizer(cls, tokenizer: object, text: str) -> tuple[int, ...]:
        kwargs = {
            "truncation": True,
            "max_length": 512,
            "padding": True,
            "return_attention_mask": True,
        }
        if hasattr(tokenizer, "batch_encode_plus"):
            encoded = tokenizer.batch_encode_plus([text], **kwargs)
        elif callable(tokenizer):
            encoded = tokenizer(text, **kwargs)
        else:
            raise ValueError("ProTrek model tokenizer is not callable")
        return cls._active_ids(encoded)

    def _fast_tokenize(self, text: str) -> TokenizedText:
        if self._offset_tokenizer is None:
            raise ValueError("tokenize_with_offsets requires frozen ProTrek paths")
        encoded = self._offset_tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        fast_ids = self._active_ids(encoded)
        offsets = tuple(
            (int(start), int(end))
            for (start, end), attention in zip(
                encoded["offset_mapping"], encoded["attention_mask"]
            )
            if int(attention) == 1
        )
        special = tuple(
            int(value)
            for value, attention in zip(
                encoded["special_tokens_mask"], encoded["attention_mask"]
            )
            if int(attention) == 1
        )
        return TokenizedText(
            text=text,
            token_ids=fast_ids,
            tokens=tuple(self._offset_tokenizer.convert_ids_to_tokens(list(fast_ids))),
            offsets=offsets,
            special_tokens_mask=special,
            attention_mask=(1,) * len(fast_ids),
            mask_token_id=int(self._offset_tokenizer.mask_token_id),
        )

    def tokenize_with_offsets(self, text: str) -> TokenizedText:
        """Tokenize with exact offsets after proving IDs match ProTrek's tokenizer."""

        if type(text) is not str:
            raise TypeError("text must be a built-in str")
        result = self._fast_tokenize(text)
        model_tokenizer = getattr(
            getattr(self.model, "text_encoder", None), "tokenizer", None
        )
        if model_tokenizer is None:
            raise ValueError("ProTrek model does not expose text_encoder.tokenizer")
        model_ids = self._ids_from_model_tokenizer(model_tokenizer, text)
        if result.token_ids != model_ids:
            raise ValueError(
                "Fast-tokenizer token-ID equivalence failed for unperturbed text"
            )
        return result

    def verify_rendered_perturbation(
        self,
        original: TokenizedText,
        rendered_text: str,
        planned_token_ids: tuple[int, ...],
    ) -> TokenizedText:
        """Prove rendered equal-length text matches planned IDs for both tokenizers."""

        if type(rendered_text) is not str or not isinstance(planned_token_ids, tuple):
            raise TypeError("rendered perturbation and planned IDs have invalid types")
        if len(planned_token_ids) != len(original.token_ids):
            raise ValueError("planned perturbation changed the non-padding token count")
        for index, (before, planned, special) in enumerate(
            zip(original.token_ids, planned_token_ids, original.special_tokens_mask)
        ):
            if special and planned != before:
                raise ValueError("planned perturbation changed a special token ID")
            if not special and planned != original.mask_token_id and planned != before:
                raise ValueError(f"planned perturbation changed non-target token {index}")
        observed = self._fast_tokenize(rendered_text)
        model_tokenizer = getattr(
            getattr(self.model, "text_encoder", None), "tokenizer", None
        )
        if model_tokenizer is None:
            raise ValueError("ProTrek model does not expose text_encoder.tokenizer")
        slow_ids = self._ids_from_model_tokenizer(model_tokenizer, rendered_text)
        if observed.token_ids != planned_token_ids or slow_ids != planned_token_ids:
            raise ValueError("rendered perturbation retokenization differs from planned IDs")
        if observed.special_tokens_mask != original.special_tokens_mask:
            raise ValueError("rendered perturbation changed special-token structure")
        return observed

    def encode(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        if not texts:
            raise ValueError("Cannot encode an empty text sequence")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                output = self.model.get_text_repr(batch)
                if output.ndim != 2 or output.shape != (len(batch), REPRESENTATION_DIM):
                    raise ValueError(
                        "ProTrek output must have shape "
                        f"({len(batch)}, {REPRESENTATION_DIM}), got {tuple(output.shape)}"
                    )
                if output.dtype != torch.float32:
                    raise ValueError(
                        f"ProTrek output must be float32, got {output.dtype}"
                    )
                if not torch.isfinite(output).all():
                    raise ValueError("ProTrek output contains non-finite values")
                norms = output.norm(dim=-1)
                if not torch.allclose(
                    norms,
                    torch.ones_like(norms),
                    atol=NORM_TOLERANCE,
                    rtol=0.0,
                ):
                    raise ValueError("ProTrek output is not L2 normalized")
                outputs.append(output.detach().cpu())
        return torch.cat(outputs, dim=0)


def load_text_embedder(
    device: str = "cuda",
    repo_root: Path = DEFAULT_PROTREK_REPO,
    weights_root: Path = DEFAULT_PROTREK_WEIGHTS,
) -> tuple[ProTrekTextEmbedder, ProTrekPaths]:
    paths = resolve_protrek_paths(repo_root, weights_root)
    model = load_protrek_model(paths, device)
    return ProTrekTextEmbedder(model, device, paths=paths), paths
