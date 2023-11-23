from time import time
import cv2
import torch
import head_segmentation.segmentation_pipeline as seg_pipeline
from prettytable import PrettyTable
import numpy as np
import os


class CustomHeadSegmentationPipeline(seg_pipeline.HumanHeadSegmentationPipeline):
    def predict(self, image: np.ndarray, name) -> np.ndarray:
        t0=time()
        print("     Predicting:")
        print("         input shape:",image.shape )
        preprocessed_image = self._preprocess_image(image)
        print("          preprocessed  input shape:", preprocessed_image.shape)

        preprocessed_image = preprocessed_image.to(self.device)

        mdl_out = self._model(preprocessed_image)

        mdl_out = mdl_out.cpu()
        print("          model_output shape", mdl_out.shape)

        # pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        pred_segmap = self._postprocess_model_output(mdl_out)
        print("         pred_segmap  shape", pred_segmap.shape)
        print("         pred_segmap unique:", np.unique(pred_segmap))
        unique, counts = np.unique(pred_segmap, return_counts=True)
        print(np.asarray((unique, counts)).T)

        return pred_segmap



def test_with_one_image(image_path, model_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("test_img shape", image.shape)

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    print("new shape", image.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)

    segmentation_pipeline_GPU = CustomHeadSegmentationPipeline(device=device, model_path=model_path)
    predicted_segmap = segmentation_pipeline_GPU.predict(image,"GPU")

    print("predicted_segmap shape:",predicted_segmap.shape)
    print("predicted_segmap unique values:",np.unique(predicted_segmap))
    return predicted_segmap


# cwd=os.getcwd()
# parent_dir=os.path.dirname(cwd)
# latest_model_path= str(parent_dir)  +'/models/last.ckpt'
latest_model_path="/home/enes/lab/training_runs/2023-11-21/15-26/models/last.ckpt"
image_path= "/home/enes/lab/processed_dataset/train/images/1001.jpg"
predicted_segmap=test_with_one_image(image_path, latest_model_path)

predicted_segmap[predicted_segmap==1]=255
cv2.imwrite("./predicted_segmap2.png", predicted_segmap)

# mask0=predicted_segmap[0,:,:]
# mask1=predicted_segmap[1,:,:]
# print(mask0.shape)
# mask0[mask0==1]
# cv2.imwrite("./mask0.png", ma