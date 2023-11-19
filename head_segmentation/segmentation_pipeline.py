import os

import cv2
import gdown
import numpy as np
import torch
from loguru import logger

import head_segmentation.constants as C
import head_segmentation.image_processing as ip
import head_segmentation.model as mdl
import torch.nn.functional as F


class HumanHeadSegmentationPipeline:
    def __init__(
        self,
        model_path: str = C.HEAD_SEGMENTATION_MODEL_PATH,
        model_url: str = C.HEAD_SEGMENTATION_MODEL_URL,
        device: torch.device = torch.device('cpu')
    ):
        if not os.path.exists(model_path):
            model_path = C.HEAD_SEGMENTATION_MODEL_PATH

            logger.warning(
                f"Model {model_path} doesn't exist. Downloading the model to {model_path}."
            )

            gdown.download(model_url, model_path, quiet=False)

        self.device = device
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        hparams = ckpt["hyper_parameters"]

        self._preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=hparams["nn_image_input_resolution"]
        )
        self._model = mdl.HeadSegmentationModel.load_from_checkpoint(
            ckpt_path=model_path
        )
        self._model.to(self.device)
        self._model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        preprocessed_image = self._preprocess_image(image)
        preprocessed_image = preprocessed_image.to(self.device)
        mdl_out = self._model(preprocessed_image)
        mdl_out = mdl_out.cpu()
        pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        return pred_segmap

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        preprocessed_image = self._preprocessing_pipeline.preprocess_image(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        return preprocessed_image

    # def _postprocess_model_output(
    #     self, out: torch.Tensor, original_image: np.ndarray
    # ) -> np.ndarray:
    #     out = out.squeeze()
    #     out = out.argmax(dim=0)
    #     out = out.numpy().astype(np.uint8)
    #     h, w = original_image.shape[:2]
    #     postprocessed = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    #
    #     return postprocessed
    # def _postprocess_model_output(
    #         self, out: torch.Tensor, original_image: np.ndarray
    # ) -> np.ndarray:
    #     # Assuming out has shape (num_classes, height, width)


        # out = torch.argmax(out, dim=0)
        #
        # out = out.numpy().astype(np.uint8)
        # print("       argmax shape", out.shape)
        # number_of_channels = out.shape[0]
        # h, w = original_image.shape[:2]
        # # postprocessed = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
        # # postprocessed = cv2.resize(out, (number_of_channels,w, h), interpolation=cv2.INTER_NEAREST)
        #
        # return out
    def _postprocess_model_output(
            self, out: torch.Tensor, original_image: np.ndarray
    ) -> np.ndarray:
        # Assuming out has shape (num_classes, height, width)

        # Print and inspect the raw model output
        print("Raw Model Output Range:", out.min().item(), out.max().item())

        # Squeeze the channel dimension
        out = out.squeeze(0)

        # Apply softmax to convert logits to probabilities
        out_probabilities = F.softmax(out, dim=0)

        # Print and inspect the unique values after softmax
        print("Model Output Probabilities Unique Values:", np.unique(out_probabilities.detach().numpy()))

        # Now use argmax on the probabilities
        predicted_classes = torch.argmax(out_probabilities, dim=0).numpy()

        # Print and inspect the unique values of predicted_classes
        print("Predicted Classes Unique Values:", np.unique(predicted_classes))

        # Check the distribution of softmax probabilities for each class
        for class_idx in range(out_probabilities.shape[0]):
            class_probs = out_probabilities[class_idx].detach().numpy()
            print(f"Class {class_idx} Probability Distribution:", np.unique(class_probs))

        # Map the predicted class values to your original class values
        class_mapping = {0: 0, 1: 1, 2: 2}  # Adjust based on your actual class mapping
        postprocessed = np.vectorize(class_mapping.get)(predicted_classes)

        # Print and inspect the unique values in the postprocessed array
        print("Postprocessed Unique Values:", np.unique(postprocessed))

        h, w = original_image.shape[:2]
        postprocessed = cv2.resize(postprocessed, (w, h), interpolation=cv2.INTER_NEAREST)

        return postprocessed
    # def _postprocess_model_output(
    #         self, out: torch.Tensor, original_image: np.ndarray
    # ) -> np.ndarray:
    #     out = out.squeeze()
    #     out_probabilities = F.softmax(out, dim=0)
    #     print("---Model Output Probabilities Unique Values:", np.unique(out_probabilities.detach().numpy()))
    #     predicted_classes = torch.argmax(out_probabilities, dim=0).numpy()
    #     print("---predicted_classes:", np.unique(predicted_classes))
    #
    #     class_mapping = {0: 0, 1: 1, 2: 2}  # Adjust based on your actual class mapping
    #     postprocessed = np.vectorize(class_mapping.get)(predicted_classes)
    #     print("---postprocessed:", np.unique(postprocessed))
    #     print("---postprocessed shape:", postprocessed.shape)
    #
    #     return postprocessed

    # def _postprocess_model_output(
    #         self, out: torch.Tensor, original_image: np.ndarray
    # ) -> np.ndarray:
    #     out = out.squeeze()
    #     out = torch.argmax(out, dim=0)
    #
    #     out = out.numpy().astype(np.uint8)
    #     print("       argmax shape", out.shape)
    #     h, w = original_image.shape[:2]
    #     postprocessed = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    #
    #     return postprocessed

