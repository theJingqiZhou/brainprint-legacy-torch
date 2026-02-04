from src.utils.registry import get_module, load_modules


def build_pipeline(config, **kwargs):
    load_modules(__file__, "pipeline")
    return get_module("pipeline", config["model"]["pipeline_name"])(config, **kwargs)
