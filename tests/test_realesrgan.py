import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from realesrgan_ncnn_py import Realesrgan
from skimage.metrics import structural_similarity

print("System version: ", sys.version)

filePATH = Path(__file__).resolve().absolute()

print("filePATH: ", filePATH)


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> bool:
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return bool(score > 0.8)


_gpuid = 0

# gpuid = -1 when in GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    _gpuid = -1

TEST_IMG = cv2.imread(str(filePATH.parent / "test.png"))


class Test_Realesrgan:
    def test_animevideov3(self) -> None:
        for _model in [0, 1, 2]:
            _realesrgan = Realesrgan(gpuid=_gpuid, model=_model)
            outimg = _realesrgan.process_cv2(TEST_IMG)

            assert calculate_image_similarity(TEST_IMG, outimg)

    @pytest.mark.skipif(_gpuid == -1, reason="skip when in GitHub Actions")
    def test_x4plus(self) -> None:
        for _model in [3, 4]:
            _realesrgan = Realesrgan(gpuid=_gpuid, model=_model)
            outimg = _realesrgan.process_cv2(TEST_IMG)

            assert calculate_image_similarity(TEST_IMG, outimg)
