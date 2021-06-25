<div align="center">
<h1>
  MMdetection + Tracker
</h1>

<h4>
  Simplest possible example of tracking. Based on <a href="https://github.com/open-mmlab/mmdetection" target="_blank">MMDetection Detector</a>.
</h4>

<a href="https://colab.research.google.com/github/fcakyon/mmdetection-object-tracker/blob/master/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://github.com/fcakyon/public-files/raw/main/mmdetection-object-tracker/vfnet-demo.gif" width="800" >

</div>

## <div align="center">Instructions</div>

1. Install Norfair with `pip install norfair[video]`.
2. Install MMDetection with `pip install torch mmcv-full mmdet`.
3. Run `python demo.py <video file>`.
4. Bonus: Use additional arguments `--model_path`, `--config_path`,`--img_scale`, `--conf_thres`, `--max_track_distance`, `--track_points` as you wish.

## <div align="center">Explanation</div>

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by MMDetection Detector.

### VFNet (MMDetection) tracking demo:

<div align="center">
<img src="https://github.com/fcakyon/public-files/raw/main/mmdetection-object-tracker/vfnet-demo.gif" width="800" >
</div>