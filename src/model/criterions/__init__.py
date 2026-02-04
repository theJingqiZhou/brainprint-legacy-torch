from src.utils.registry import get_module, load_modules


def build_criterions(config):
    load_modules(__file__, "criterions")
    return get_module("criterions", config["train"]["criterions"])(config)
