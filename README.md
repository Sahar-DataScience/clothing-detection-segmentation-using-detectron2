# clothing-detection-segmentation-using-detectron2
cutomizing mask rcnn backboned with FPN + Resnet50 from detecton2 model zoo on deep fashion2 dataset
<!--<p align='center'>-->
<img src='https://raw.githubusercontent.com/Sahar-DataScience/clothing-detection-segmentation-using-detectron2/main/bigbang.png' width='65%'/>
<!--![My Image](bigbang.png)-->

## 1. Dataset 
the model was trained on 50k images extracted from [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) which is a comprehensive fashion dataset. It contains 191K diverse images of 13 popular clothing categories from both commercial shopping stores and consumers. Each item in an image is labeled with scale, occlusion, zoom-in, viewpoint, category, style, bounding box, dense landmarks and per-pixel mask.

###  - Dataset Preprocessing 
the extracted 50k images considered as good data where each item is labeled with low occlusion, medium scale and acceptable zoom in 
the number of classes where reduced from 13 to only 5 categories : short sleeved shirt, long sleeved shirt , outwear , shorts and trousers in order to make new balanced data set to ameliorate training results

<img src='https://raw.githubusercontent.com/Sahar-DataScience/clothing-detection-segmentation-using-detectron2/main/images/five_classes.png' width='50%'/>

### - final training set :
```
[04/04 11:33:20 d2.data.build]: Removed 0 images with no usable annotations. 42531 images left.
[04/04 11:33:22 d2.data.build]: Distribution of instances among all 5 categories:
|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
| short_sleev.. | 18034        | long_sleeve.. | 14072        | long_sleeve.. | 9559         |
|    shorts     | 11752        |   trousers    | 17430        |               |              |
|     total     | 70847        |               |              |               |              |
```

## 2. Model
Mask RCNN backboned with Resnet 50 and Feature Pyramid Network (FPN) pretrained on coco dataset, selected from [Detectron2 Medel zoo]( https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
```
mask_rcnn_R_50_FPN_3x.yaml
```

### - Environment Set up
you can follow installation guide in this [blog](https://medium.com/@muhammadtalha1726/detectron-2-installation-guide-d12f66e220bf)
```
conda create -n DeepFashion python=3.6
conda activate DeepFashion
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install pycocotools
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

```
### - Training details
      -  The pretrained model  modelCOCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x   (box AP 41 Mask AP 37)
      -  2 backbone layers are freezed to avoid overfitting
      -  Total iterations 271k
      -  LR 0.001 reduced at iteration 163k and 230k
      -  Checkpoints saved each 13k iterations 
      -  Training on 54k images 
      
to understand more about detetcron2 hyperparameters configuration check [this](https://detectron2.readthedocs.io/en/latest/modules/config.html?highlight=config)
## 3. Evaluation
### - Test set 
```
[04/05 15:36:56 d2.data.build]: Distribution of instances among all 5 categories:
|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
| short_sleev.. | 4615         | long_sleeve.. | 3615         | long_sleeve.. | 2640         |
|    shorts     | 3072         |   trousers    | 4519         |               |              |
|     total     | 18461        |               |              |               |              |
```
### - Coco evaluation metrics
```
Evaluate annotation type *bbox*
[04/05 15:48:50 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 1.19 seconds.
[04/05 15:48:50 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/05 15:48:50 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.13 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.840
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.978
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.953
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.900
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.776
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.841
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.878
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.891
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.891
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.900
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.821
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.891
[04/05 15:48:50 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 84.014 | 97.783 | 95.311 | 90.000 | 77.562 | 84.080 |
[04/05 15:48:50 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category            | AP     | category           | AP     | category             | AP     |
|:--------------------|:-------|:-------------------|:-------|:---------------------|:-------|
| short_sleeved_shirt | 86.096 | long_sleeved_shirt | 84.978 | long_sleeved_outwear | 87.261 |
| shorts              | 80.733 | trousers           | 81.004 |                      |        |

Loading and preparing results...
DONE (t=0.22s)
creating index...
index created!
[04/05 15:48:50 d2.evaluation.fast_eval_api]: Evaluate annotation type *segm*
[04/05 15:48:54 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 3.18 seconds.
[04/05 15:48:54 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/05 15:48:54 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.12 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.792
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.976
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.943
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.767
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.837
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.837
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.747
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.838
[04/05 15:48:54 d2.evaluation.coco_evaluation]: Evaluation results for segm:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 79.205 | 97.589 | 94.333 | 76.691 | 62.900 | 79.356 |
[04/05 15:48:54 d2.evaluation.coco_evaluation]: Per-category segm AP:
| category            | AP     | category           | AP     | category             | AP     |
|:--------------------|:-------|:-------------------|:-------|:---------------------|:-------|
| short_sleeved_shirt | 86.406 | long_sleeved_shirt | 81.597 | long_sleeved_outwear | 69.072 |
| shorts              | 79.815 | trousers           | 79.133 |                      |        |
OrderedDict([('bbox', {'AP': 84.01428995724827, 'AP50': 97.78271739220436, 'AP75': 95.31086500403688, 'APs': 90.0, 'APm': 77.56184022471318, 'APl': 84.08033907097128, 'AP-short_sleeved_shirt': 86.09565841415512, 'AP-long_sleeved_shirt': 84.97832571093082, 'AP-long_sleeved_outwear': 87.2606139252723, 'AP-shorts': 80.7328714070072, 'AP-trousers': 81.00398032887594}), ('segm', {'AP': 79.20463838317468, 'AP50': 97.58865617295218, 'AP75': 94.3327826617389, 'APs': 76.69141914191418, 'APm': 62.89989410886051, 'APl': 79.35623090795374, 'AP-short_sleeved_shirt': 86.40581702012908, 'AP-long_sleeved_shirt': 81.5966939957104, 'AP-long_sleeved_outwear': 69.07223653143465, 'AP-shorts': 79.81532974646667, 'AP-trousers': 79.13311462213261})])
```
### - Average Precision per class 

0 < AP < 1

<img src='https://raw.githubusercontent.com/Sahar-DataScience/clothing-detection-segmentation-using-detectron2/main/images/AP_per_class.PNG' width='50%'/>
