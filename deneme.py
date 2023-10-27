
# image_path= "/home/enes/lab/head-segmentation/hb_img1.png"

from time import time
import cv2
import torch
import head_segmentation.segmentation_pipeline as seg_pipeline
from prettytable import PrettyTable
import numpy as np

class CustomHeadSegmentationPipeline(seg_pipeline.HumanHeadSegmentationPipeline):
    def predict(self, image: np.ndarray, name) -> np.ndarray:
        t0=time()
        preprocessed_image = self._preprocess_image(image)
        t1 = time()
        preprocessed_image = preprocessed_image.to(self.device)
        t2 = time()
        mdl_out = self._model(preprocessed_image)
        t3 = time()
        mdl_out = mdl_out.cpu()
        t4 = time()
        pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        t5= time()

        print(" ")
        print("Test details for :", name)
        print(" ")

        print("preprocessing",round(t1-t0,3))
        print("to cpu/gpu",round(t2-t1,3))
        print("model output",round(t3-t2,3))
        print("to cpu",round(t4-t3,3))
        print("postprocess",round(t5-t4,3))
        print("total",round(t5-t0,3))
        print("-------------")

        return pred_segmap


print("----Loading Test images----")
#img path for one of orignal celebA images (1024x1024)
image_path= "/home/enes/lab/processed_dataset/test/images/1000.jpg"
# image_path= "/home/enes/lab/head-segmentation/processed_dataset/test/images/1000.jpg"


image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("test_img shape", image.shape)

image_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
image_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
print("resized_test_img (512,512) shape", image_512.shape)
print("resized_test_img (256,256) shape", image_256.shape)

print("----    ----")
print("  ")

print("----Check if GPU is available----")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
print("----    ----")
print("  ")


model_path_mobilenet_v2= "/home/enes/lab/head-segmentation/training_runs/2023-10-22/00-16/models/last.ckpt"
model_path_resnet34= "/home/enes/lab/head-segmentation/training_runs/2023-10-22/21-22/models/last.ckpt"
model_path_resnet34_256="/home/enes/lab/head-segmentation/training_runs/2023-10-26/18-54/models/last.ckpt"


model_path=model_path_resnet34_256



segmentation_pipeline = CustomHeadSegmentationPipeline(model_path=model_path)
segmentation_pipeline_GPU = CustomHeadSegmentationPipeline(device=device, model_path=model_path)

t0=time()
name="1024 + CPU"
predicted_segmap = segmentation_pipeline.predict(image, name)
t1=time()
name="512 + CPU"
predicted_segmap = segmentation_pipeline.predict(image_512, name)
t2=time()
name="216 + CPU"
predicted_segmap = segmentation_pipeline.predict(image_256,name)
t3=time()
name="1024 + GPU"
predicted_segmap = segmentation_pipeline_GPU.predict(image,name)
t4=time()
name="512 + GPU"
predicted_segmap = segmentation_pipeline_GPU.predict(image_512, name)
t5=time()
name="256 + GPU"
predicted_segmap = segmentation_pipeline_GPU.predict(image_256, name)
t6=time()


print("Inference times for resnet34 --pretrained --depth=3 : ")
myTable = PrettyTable(["Image Size", "CPU", "GPU", ])

myTable.add_row(["1024", str(round(t1-t0,2))+" sec", str(round(t4-t3,2))+" sec"])
myTable.add_row(["512", str(round(t2-t1,2))+" sec", str(round(t5-t4,2))+" sec"])
myTable.add_row(["256", str(round(t3-t2,2))+" sec", str(round(t6-t5,2))+" sec"])

print(myTable)
