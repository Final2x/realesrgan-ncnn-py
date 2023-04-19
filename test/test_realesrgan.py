import sys
import os
import time
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
import cv2
import pytest
from pathlib import Path

try:
    from realesrgan_ncnn_vulkan import Realesrgan
except ImportError:
    from realesrgan_ncnn_py import Realesrgan


def calculate_image_similarity() -> bool:
    # Load the two images
    image1 = cv2.imread("./test.png")
    image2 = cv2.imread("./output.png")
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return score > 0.5


model_dict = {
    0: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2},
    1: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3},
    2: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4},
    3: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4},
    4: {"param": "realesrgan-x4plus.param", "bin": "realesrgan-x4plus.bin", "scale": 4}
}

print("System version: ", sys.version)

_gpuid = 0


class Test_Realesrgan:
    def test_pil(self):
        for i in range(5):
            _scale = model_dict[i]["scale"]
            with Image.open("test.png") as image:
                out_w = image.width * _scale
                out_h = image.height * _scale
                _realesrgan = Realesrgan(gpuid=_gpuid, model=i)
                image = _realesrgan.process_pil(image)
                image.save("output.png")

            assert calculate_image_similarity() == True

            with Image.open("output.png") as image:
                assert image.width == out_w
                assert image.height == out_h

    def test_cv2(self):
        for i in range(5):
            _scale = model_dict[i]["scale"]
            image = cv2.imdecode(np.fromfile("test.png", dtype=np.uint8), cv2.IMREAD_COLOR)

            out_w = image.shape[1] * _scale
            out_h = image.shape[0] * _scale

            _realesrgan = Realesrgan(gpuid=_gpuid, model=i)
            image = _realesrgan.process_cv2(image)
            cv2.imencode(".jpg", image)[1].tofile("output.png")

            assert calculate_image_similarity() == True

            image = cv2.imdecode(np.fromfile("output.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image.shape[1] == out_w
            assert image.shape[0] == out_h

    def test_custom(self):
        _realesrgan = Realesrgan(gpuid=_gpuid, model=-1)

        with pytest.raises(ValueError) as e:
            _realesrgan.load()
        assert e.value.args[0] == "param_path, model_path and scale must be specified when model == -1"

        with pytest.raises(ValueError) as e:
            _realesrgan.load(param_path=Path("/user/documents/models/realesr-animevideov3-x2.param"))
        assert e.value.args[0] == "param_path and model_path must be specified when model == -1"

        with pytest.raises(ValueError) as e:
            _realesrgan.load(param_path=Path("/user/documents/models/realesr-animevideov3-x2.param"),
                             model_path=Path("/user/documents/models/realesr-animevideov3-x2.bin"))
        assert e.value.args[0] == "scale must be specified when model == -1"
