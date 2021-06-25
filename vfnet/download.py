import urllib.request
from pathlib import Path

FILE = Path(__file__).parent.absolute()


def run():
    urllib.request.urlretrieve(
        "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth",
        str(FILE / "vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth"),
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/fcakyon/mmdetection-object-tracker/master/vfnet/vfnet_r50_fpn_mstrain_2x_coco.py",
        str(FILE / "vfnet_r50_fpn_mstrain_2x_coco.py"),
    )


if __name__ == "__main__":
    run()
