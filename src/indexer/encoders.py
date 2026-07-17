import torch
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def norm_path(stored_path):
    """Normalize a stored dataset path to the current OS separator.

    The metadata CSVs were written on Windows (``data\\raw\\img_0000.jpg``);
    on POSIX the backslashes would otherwise be treated as part of the
    filename. Kept here so every builder/retriever resolves paths the same way.
    """
    return Path(str(stored_path).replace("\\", "/"))

class CLIPEncoder:
    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        )

        self.tokenizer = open_clip.get_tokenizer(
            model_name
        )

        self.model = self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def normalize(embeddings):
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.clip(norms, 1e-12, None)

        return embeddings / norms

    def encode_images(self, image_paths):
        images = []

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")

            image = self.preprocess(image)

            images.append(image)

        image_batch = torch.stack(images).to(
            self.device
        )

        with torch.inference_mode():
            embeddings = self.model.encode_image(
                image_batch
            )

        embeddings = (
            embeddings
            .float()
            .cpu()
            .numpy()
        )

        return self.normalize(embeddings).astype(
            np.float32
        )

    def encode_texts(self, texts):
        tokens = self.tokenizer(texts).to(
            self.device
        )

        with torch.inference_mode():
            embeddings = self.model.encode_text(
                tokens
            )

        embeddings = (
            embeddings
            .float()
            .cpu()
            .numpy()
        )

        return self.normalize(embeddings).astype(
            np.float32
        )
   



class FashionCLIPEncoder:
    def __init__(
        self,
        model_name="patrickjohncyh/fashion-clip",
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")
        print(f"Loading FashionCLIP: {model_name}")

        self.model = CLIPModel.from_pretrained(
            model_name
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

        self.model.eval()

    @staticmethod
    def normalize(embeddings):
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.clip(
            norms,
            1e-12,
            None,
        )

        return embeddings / norms

    def encode_images(self, image_paths, batch_size=32):
        images = []

        for image_input in image_paths:
            if isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                image = Image.open(
                    image_input
                ).convert("RGB")

            images.append(image)

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(self.device)

        # get_image_features applies the visual projection internally and is
        # stable across transformers versions (the manual vision_model(...,
        # return_dict=True) + visual_projection path broke in transformers
        # >=4.5x, where submodule forward() no longer accepts return_dict).
        with torch.inference_mode():
            embeddings = self.model.get_image_features(
                pixel_values=pixel_values,
            )

        # transformers >=5 returns BaseModelOutputWithPooling whose
        # pooler_output is the projected feature; <5 returns the tensor.
        if not torch.is_tensor(embeddings):
            embeddings = embeddings.pooler_output

        embeddings = (
            embeddings
            .float()
            .cpu()
            .numpy()
        )

        return self.normalize(
            embeddings
        ).astype(np.float32)

    def encode_texts(self, texts):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inputs[
            "input_ids"
        ].to(self.device)

        attention_mask = inputs[
            "attention_mask"
        ].to(self.device)

        # get_text_features applies the text projection internally and is
        # stable across transformers versions (see encode_images note).
        with torch.inference_mode():
            embeddings = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if not torch.is_tensor(embeddings):
            embeddings = embeddings.pooler_output

        embeddings = (
            embeddings
            .float()
            .cpu()
            .numpy()
        )

        return self.normalize(
            embeddings
        ).astype(np.float32)