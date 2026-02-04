from src.utils.registry import get_module, load_modules


def build_backbone(config):
    load_modules(__file__, "backbone")
    return get_module("backbone", config["model"]["backbone_name"])(config)
