# segmentation_share

### install packages
1. python3 -m pip install tensorflow==1.14

### source
1. segmentation
*   https://github.com/tensorflow/models/tree/master/research/deeplab
*   https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
2. matting
*   https://github.com/huochaitiantang/pytorch-deep-image-matting
*   git clone https://github.com/huochaitiantang/pytorch-deep-image-matting
*   cp -avr pytorch-deep-image-matting/result .
*   cp pytorch-deep-image-matting/core/deploy.py .
*   cp pytorch-deep-image-matting/core/net.py .
*   rm -rf result/example/pred
*   wget https://github.com/huochaitiantang/pytorch-deep-image-matting/releases/download/v1.4/stage1_sad_54.4.pth
3. trimap
*   https://github.com/lnugraha/trimap_generator
*   git clone https://github.com/lnugraha/trimap_generator/
*   cp -avr trimap_generator/images/test_images .
*   from PIL import Image
*   Image.fromarray
### step 
1. git clone https://github.com/jack155861/segmentation_share 
2. cd segmentation_share
3. wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz
4. wget https://github.com/huochaitiantang/pytorch-deep-image-matting/releases/download/v1.4/stage1_sad_54.4.pth
