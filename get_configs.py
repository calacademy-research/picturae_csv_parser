"""docstring: stores the Config Loader, used to retrieve python config files """
import warnings
import os
from importlib import util
# from collection_definitions import COLLECTION_DIRS

COLLECTION_DIRS = {
    # 'COLLECTION_NAME': 'DIRECTORY_NAME',
    'Botany': 'botany',
    'Botany_PIC': 'picturae_csv'
}

def get_config(config: str):
    """reads in collections module from python config file"""
    current_dir = os.getcwd()
    if config in COLLECTION_DIRS.keys() or config == "picbatch":
        pass
    else:
        warnings.warn(f"undefined collection for {config} in collection definitions")

    config = config.lower()

    location = f"config_files/{config}_config.py"

    try:
        if os.path.exists(os.path.join(current_dir, "picturae_csv_create.py")):
            pass
        else:
            location = os.path.join("..", location)

        spec = util.spec_from_file_location(name=f"{config}_config",
                                            location=location)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        warnings.warn(f"File for collection {config} is missing. "
                      f"Please check that the configuration file is present")
        return None
