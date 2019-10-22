import os
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# DeepLabV3+ model loading
class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513 #圖片長寬
    FROZEN_GRAPH_NAME = 'frozen' #_inference_graph
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()  
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
MODEL_xception65_trainval = DeepLabModel("model_xception65_coco_voc_trainval.tar.gz")
print("deeplabv3+ model loading")

    
def deeplabv3plus(photo_input):
    MODEL = MODEL_xception65_trainval
    original_im = Image.open(photo_input)
    width, height = original_im.size
    resized_im, seg_map = MODEL.run(original_im)
    cm = seg_map
    img = np.array(resized_im)
    rows = cm.shape[0]
    cols = cm.shape[1]
    
    img_seq = img.copy()
    img_seq[cm==0] = np.array([255, 255,255], dtype='uint8') 
    img_seq = Image.fromarray(img_seq).resize((width, height),Image.ANTIALIAS) 
    
    img_mask = img.copy()
    img_mask[cm==0] = np.array([0, 0,0], dtype='uint8') 
    img_mask[cm!=0] = np.array([255, 255,255], dtype='uint8') 
    img_mask = Image.fromarray(img_mask).resize((width, height),Image.ANTIALIAS)

    return img_seq, img_mask
    