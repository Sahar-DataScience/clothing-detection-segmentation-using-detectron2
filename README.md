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

