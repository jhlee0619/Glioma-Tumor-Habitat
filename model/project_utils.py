"""Minimal instantiation helper used by encode.py to load the deposited dVAE.

`instantiate_from_config` receives an OmegaConf node with a `target` string
(dotted import path) and optional `params` dict, imports the target class,
and instantiates it. This is the only helper needed to reload the model
from the deposited checkpoint.
"""
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
