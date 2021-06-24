import urllib.request
from os import path
from pathlib import Path
from importlib import import_module
import shutil
import sys
from typing import Optional, Tuple


MODEL_NAME2MODEL_URL: dict = {
    "cascade_rcnn": "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth",
    "retinanet": "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth",
}


def parse_model_name_and_config_name_from_model_url(url: str):
    model_name = url.split("/")[5]
    config_name = url.split("/")[-2] + ".py"
    return model_name, config_name


class MmdetPretrainedModel:
    def __init__(self, model_url: Optional[str] = None, model_name: Optional[str] = None):
        # if model_name is provided, parse model_url from MODEL_NAME2MODEL_URL mapping
        if model_name:
            assert (
                model_name in MODEL_NAME2MODEL_URL.keys()
            ), f"model_name should be one of {list(MODEL_NAME2MODEL_URL.keys())} but given as {model_name}"
            model_url = MODEL_NAME2MODEL_URL[model_name]

        assert model_url, "you have to pass either model_url or model_name"

        # set MODEL_URL, MODEL_NAME, CONFIG_NAME
        self.MODEL_URL = model_url
        self.MODEL_NAME, self.CONFIG_NAME = parse_model_name_and_config_name_from_model_url(model_url)

    def download(self, destination_dir: str = "./") -> Tuple[str, str]:
        # create destination_dir and set model_path
        model_path = Path(destination_dir) / Path(self.MODEL_URL).name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = str(model_path)
        config_path = str(Path(destination_dir) / self.CONFIG_NAME)

        # download checkpoint if not present
        if not path.exists(model_path):
            urllib.request.urlretrieve(
                self.MODEL_URL,
                model_path,
            )

        # downlaod model config if not present
        if not path.exists(config_path):
            config_path = download_mmdet_config(
                destination_dir=destination_dir,
                model_name=self.MODEL_NAME,
                config_file_name=self.CONFIG_NAME,
                verbose=True,
            )

        return model_path, config_path


def download_mmdet_config(
    destination_dir: str = "./",
    model_name: str = "cascade_rcnn",
    config_file_name: str = "cascade_mask_rcnn_r50_fpn_1x_coco.py",
    verbose: bool = True,
) -> str:
    """
    Merges config files starting from given main config file name. Saves as single file.
    Args:
        model_name (str): mmdet model name. check https://github.com/open-mmlab/mmdetection/tree/master/configs.
        config_file_name (str): mdmet config file name.
        verbose (bool): if True, print save path.
    Returns:
        (str) abs path of the downloaded config file.
    """

    # get mmdet version
    from mmdet import __version__

    mmdet_ver = "v" + __version__

    # set main config url
    base_config_url = (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/" + mmdet_ver + "/configs/" + model_name + "/"
    )
    main_config_url = base_config_url + config_file_name

    # set config dirs
    temp_configs_dir = Path("temp_mmdet_configs")
    main_config_dir = temp_configs_dir / model_name

    # create config dirs
    temp_configs_dir.mkdir(parents=True, exist_ok=True)
    main_config_dir.mkdir(parents=True, exist_ok=True)

    # get main config file name
    filename = Path(main_config_url).name

    # set main config file path
    main_config_path = str(main_config_dir / filename)

    # download main config file
    urllib.request.urlretrieve(
        main_config_url,
        main_config_path,
    )

    # read main config file
    sys.path.insert(0, str(main_config_dir))
    temp_module_name = path.splitext(filename)[0]
    mod = import_module(temp_module_name)
    sys.path.pop(0)
    config_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    # iterate over secondary config files
    config_list = config_dict["_base_"] if type(config_dict["_base_"]) == list else [config_dict["_base_"]]
    for secondary_config_file_path in config_list:
        # set config url
        config_url = base_config_url + secondary_config_file_path
        config_path = main_config_dir / secondary_config_file_path

        # create secondary config dir
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # download secondary config files
        urllib.request.urlretrieve(
            config_url,
            str(config_path),
        )

        # read secondary config file
        sys.path.insert(0, str(main_config_dir))
        filename = Path(config_path).name
        temp_module_name = path.splitext(filename)[0]
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        config_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
        try:
            # iterate over secondary config files
            config_list = config_dict["_base_"] if type(config_dict["_base_"]) == list else [config_dict["_base_"]]
            for secondary_config_file_path in config_list:
                # set config url
                config_url = base_config_url + secondary_config_file_path
                config_path = main_config_dir / secondary_config_file_path

                # create secondary config dir
                config_path.parent.mkdir(parents=True, exist_ok=True)

                # download secondary config files
                urllib.request.urlretrieve(
                    config_url,
                    str(config_path),
                )
        except:
            pass

    # get final config file name
    filename = Path(main_config_url).name

    # set final config file path
    final_config_path = str(Path(destination_dir) / filename)

    # dump final config as single file
    from mmcv import Config

    config = Config.fromfile(main_config_path)
    config.dump(final_config_path)

    if verbose:
        print(f"mmdet config file has been downloaded to {path.abspath(final_config_path)}")

    # remove temp config dir
    shutil.rmtree(temp_configs_dir)

    return path.abspath(final_config_path)
