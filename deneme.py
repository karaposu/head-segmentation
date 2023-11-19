#
# # image_path= "/home/enes/lab/head-segmentation/hb_img1.png"
#
from time import time
import cv2
import torch
import head_segmentation.segmentation_pipeline as seg_pipeline
from prettytable import PrettyTable
import numpy as np
import os

print(torch.__version__)

def get_latest_model_checkpoint_path(parent_dir):
    file_name = 'models/last.ckpt'
    date_folders = os.listdir(parent_dir)
    latest_checkpoint_path = None

    for date_folder in date_folders:
        time_folders = os.listdir(os.path.join(parent_dir, date_folder))
        time_folders_with_file = [time_folder for time_folder in time_folders if
                                  os.path.isfile(os.path.join(parent_dir, date_folder, time_folder, file_name))]
        sorted_time_folders_with_file = sorted(time_folders_with_file, reverse=True)
        if sorted_time_folders_with_file:
            latest_checkpoint_path = os.path.join(parent_dir, date_folder, sorted_time_folders_with_file[0])
            print("latest_checkpoint_path:", latest_checkpoint_path)
            print("date_folder:", date_folder)
            print("sorted_time_folders_with_file[0]:", sorted_time_folders_with_file[0])
            latest_checkpoint_path = latest_checkpoint_path + "/"+file_name
            break
        # latest_checkpoint_path=latest_checkpoint_path+file_name
    return latest_checkpoint_path


image_path= "/home/enes/lab/processed_dataset/test/images/1000.jpg"

# training_runs_dir="/home/enes/lab/head-segmentation/training_runs"
# training_runs_dir="/home/enes/lab/training_runs/2023-11-14/13-22/models"

training_runs_dir="/home/enes/lab/training_runs"


latest_model_path= get_latest_model_checkpoint_path(training_runs_dir)
print("latest_model_path:", latest_model_path)

model_path_mobilenet_v2= "/home/enes/lab/training_runs/2023-10-22/00-16/models/last.ckpt"
model_path_resnet34= "/home/enes/lab/training_runs/2023-10-22/21-22/models/last.ckpt"
model_path_resnet34_256="/home/enes/lab/training_runs/2023-10-26/18-54/models/last.ckpt"

model_path=latest_model_path

class CustomHeadSegmentationPipeline(seg_pipeline.HumanHeadSegmentationPipeline):
    def predict(self, image: np.ndarray, name) -> np.ndarray:
        t0=time()
        preprocessed_image = self._preprocess_image(image)
        print("preprocessed img shape",preprocessed_image.shape)
        t1 = time()
        preprocessed_image = preprocessed_image.to(self.device)
        t2 = time()
        mdl_out = self._model(preprocessed_image)
        t3 = time()
        mdl_out = mdl_out.cpu()
        print("model output shape", mdl_out.shape)
        # print("mdl_out unique pixels:", np.unique(mdl_out))

        t4 = time()
        pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        print("pred_segmap  shape", pred_segmap.shape)
        print("pred_segmap unique:", np.unique(pred_segmap))
        # z = np.unique(mdl_out.detach().numpy())
        # print(" unique pixels:", np.unique(z), "len:", len(np.unique(z)) )
        # print("postprocessed  shape", pred_segmap.shape)


        # print(" channle0 unique pixels:", np.unique(pred_segmap[:, :, 0]))
        # print(" channle1 unique pixels:", np.unique(pred_segmap[:, :, 1]))
        # print(" channle2 unique pixels:", np.unique(pred_segmap[:, :, 2]))
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

# image_path= "/home/enes/lab/head-segmentation/processed_dataset/test/images/1000.jpg"


image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("test_img shape", image.shape)

image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
print("new shape", image.shape)

print("----    ----")
print("  ")

print("----Check if GPU is available----")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
print("----    ----")
print("  ")


segmentation_pipeline = CustomHeadSegmentationPipeline(model_path=model_path)
segmentation_pipeline_GPU = CustomHeadSegmentationPipeline(device=device, model_path=model_path)

t0=time()
name="216 + CPU"
predicted_segmap = segmentation_pipeline.predict(image,name)
t1=time()
predicted_segmap = segmentation_pipeline.predict(image,name)
t2=time()

name="256 + GPU"
predicted_segmap = segmentation_pipeline_GPU.predict(image,name)
t3=time()
predicted_segmap = segmentation_pipeline_GPU.predict(image, name)
t4=time()

print(predicted_segmap.shape)
print(np.unique(predicted_segmap))

predicted_segmap[predicted_segmap==1]=255
predicted_segmap = np.uint8(predicted_segmap)
# print(np.unique(predicted_segmap))

unique_values, counts = np.unique(predicted_segmap, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"Frequency of {value}: {count}")



cv2.imwrite("./semap.png", predicted_segmap)


print("Inference times for resnet34 --pretrained --depth=3 : ")
myTable = PrettyTable(["Image Size", "CPU", "GPU", ])

myTable.add_row(["256", str(round(t1-t0,2))+" sec", str(round(t3-t2,2))+" sec"])
myTable.add_row(["256", str(round(t2-t1,2))+" sec", str(round(t4-t3,2))+" sec"])


print(myTable)
