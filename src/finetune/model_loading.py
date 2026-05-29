import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model.kronos import Kronos, KronosTokenizer


WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
HF_ENDPOINTS = tuple(
    endpoint.rstrip("/")
    for endpoint in (
        os.environ.get("HF_ENDPOINT"),
        "https://huggingface.co",
        "https://hf-mirror.com",
    )
    if endpoint
)


TOKENIZER_CONFIGS = {
    "NeoQuasar/Kronos-Tokenizer-base": {
        "d_in": 6,
        "d_model": 256,
        "n_heads": 4,
        "ff_dim": 512,
        "n_enc_layers": 4,
        "n_dec_layers": 4,
        "ffn_dropout_p": 0.0,
        "attn_dropout_p": 0.0,
        "resid_dropout_p": 0.0,
        "s1_bits": 10,
        "s2_bits": 10,
        "beta": 0.05,
        "gamma0": 1.0,
        "gamma": 1.1,
        "zeta": 0.05,
        "group_size": 4,
    }
}


PREDICTOR_CONFIGS = {
    "NeoQuasar/Kronos-small": {
        "s1_bits": 10,
        "s2_bits": 10,
        "n_layers": 8,
        "d_model": 512,
        "n_heads": 8,
        "ff_dim": 1024,
        "ffn_dropout_p": 0.25,
        "attn_dropout_p": 0.1,
        "resid_dropout_p": 0.25,
        "token_dropout_p": 0.1,
        "learn_te": True,
    }
}


def _is_missing_init_args_error(error: TypeError) -> bool:
    return "missing" in str(error) and "required positional arguments" in str(error)


def _load_json(path_or_repo: str, fallback: dict | None) -> dict:
    local_config = Path(path_or_repo) / "config.json"
    if local_config.exists():
        with local_config.open("r", encoding="utf-8") as f:
            return json.load(f)

    errors = []
    for endpoint in HF_ENDPOINTS:
        try:
            config_path = hf_hub_download(path_or_repo, "config.json", endpoint=endpoint)
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as error:
            errors.append(f"{endpoint}: {type(error).__name__}: {error}")

    if fallback is not None:
        return fallback.copy()

    raise FileNotFoundError(
        f"Could not load config.json for {path_or_repo}. Tried endpoints: "
        + "; ".join(errors)
    )


def _hub_download(path_or_repo: str, filename: str) -> str:
    errors = []
    for endpoint in HF_ENDPOINTS:
        try:
            return hf_hub_download(path_or_repo, filename, endpoint=endpoint)
        except Exception as error:
            errors.append(f"{endpoint}: {type(error).__name__}: {error}")

    raise FileNotFoundError(
        f"Could not download {filename} for {path_or_repo}. Tried endpoints: "
        + "; ".join(errors)
    )


def _download_any_weight(path_or_repo: str) -> str:
    errors = []
    for filename in WEIGHT_FILENAMES:
        try:
            return _hub_download(path_or_repo, filename)
        except FileNotFoundError as error:
            errors.append(str(error))

    raise FileNotFoundError(
        f"No model weights found for {path_or_repo}. Tried filenames "
        f"{', '.join(WEIGHT_FILENAMES)}. Details: "
        + " | ".join(errors)
    )


def _find_local_weights(path_or_repo: str) -> str:
    local_dir = Path(path_or_repo)
    for filename in WEIGHT_FILENAMES:
        weight_path = local_dir / filename
        if weight_path.exists():
            return str(weight_path)

    raise FileNotFoundError(
        f"No model weights found in {path_or_repo}. Expected one of: "
        + ", ".join(WEIGHT_FILENAMES)
    )


def _resolve_weights(path_or_repo: str) -> str:
    if Path(path_or_repo).is_dir():
        return _find_local_weights(path_or_repo)

    return _download_any_weight(path_or_repo)


def _load_state_dict(weight_path: str) -> dict:
    if weight_path.endswith(".safetensors"):
        return load_file(weight_path)
    return torch.load(weight_path, map_location="cpu")


def _manual_load(cls, path_or_repo: str, fallback_config: dict | None):
    model_config = _load_json(path_or_repo, fallback_config)
    model = cls(**model_config)
    weight_path = _resolve_weights(path_or_repo)
    state_dict = _load_state_dict(weight_path)
    model.load_state_dict(state_dict)
    return model


def load_tokenizer(path_or_repo: str) -> KronosTokenizer:
    try:
        return KronosTokenizer.from_pretrained(path_or_repo)
    except TypeError as error:
        if not _is_missing_init_args_error(error):
            raise
        return _manual_load(KronosTokenizer, path_or_repo, TOKENIZER_CONFIGS.get(path_or_repo))


def load_predictor(path_or_repo: str) -> Kronos:
    try:
        return Kronos.from_pretrained(path_or_repo)
    except TypeError as error:
        if not _is_missing_init_args_error(error):
            raise
        return _manual_load(Kronos, path_or_repo, PREDICTOR_CONFIGS.get(path_or_repo))
