"""
The MIT License (MIT)

Copyright (c) 2021 ArchieMeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# 参考https://github.com/media2x/realsr-ncnn-vulkan-python, 感谢原作者

import pathlib
from typing import Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image

try:
    from . import realesrgan_ncnn_vulkan_wrapper as wrapped
except ImportError:
    import realesrgan_ncnn_vulkan_wrapper as wrapped


class Realesrgan:
    def __init__(self, gpuid: int = 0, tta_mode: bool = False, tilesize: int = 0, model: int = 0):
        """
        RealESRGAN class for Super Resolution

        :param gpuid: gpu device to use, -1 for cpu
        :param tta_mode: enable test time argumentation
        :param tilesize: tile size, 0 for auto, must >= 32
        :param model: realesrgan model, 0 for default, -1 for custom load
        """

        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert model >= -1, "model must > 0 or -1"

        self._gpuid = gpuid

        self._realesrgan_object = wrapped.RealESRGANWrapped(gpuid, tta_mode)

        self._tilesize = tilesize
        self._model = model
        self._scale = 2

        if self._model > -1:
            self._load()

        self.raw_in_image = None
        self.raw_out_image = None

    def _set_parameters(self) -> None:
        """
        Set parameters for RealESRGAN

        :return: None
        """
        self._realesrgan_object.set_parameters(self._tilesize, self._scale)

    def _load(
        self, param_path: Optional[pathlib.Path] = None, model_path: Optional[pathlib.Path] = None, scale: int = 0
    ) -> None:
        """
        Load models from given paths when self._model == -1

        :param param_path: the path to model params. usually ended with ".param"
        :param model_path: the path to model bin. usually ended with ".bin"
        :param scale: the scale of the model. 1, 2, 3, 4...
        :return: None
        """

        model_dict: Dict[int, Dict[str, Union[str, int]]] = {
            0: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2},
            1: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3},
            2: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4},
            3: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4},
            4: {"param": "realesrgan-x4plus.param", "bin": "realesrgan-x4plus.bin", "scale": 4},
        }

        if self._model == -1:
            if param_path is None and model_path is None and scale == 0:
                raise ValueError("param_path, model_path and scale must be specified when model == -1")
            if param_path is None or model_path is None:
                raise ValueError("param_path and model_path must be specified when model == -1")
            if scale == 0:
                raise ValueError("scale must be specified when model == -1")
        else:
            model_dir = pathlib.Path(__file__).parent / "models"

            param_path = model_dir / pathlib.Path(str(model_dict[self._model]["param"]))
            model_path = model_dir / pathlib.Path(str(model_dict[self._model]["bin"]))

        self._scale = scale if scale != 0 else int(model_dict[self._model]["scale"])
        self._set_parameters()

        if param_path is None or model_path is None:
            raise ValueError("param_path and model_path is None")

        self._realesrgan_object.load(str(param_path), str(model_path))

    def process(self) -> None:
        self._realesrgan_object.process(self.raw_in_image, self.raw_out_image)

    def process_pil(self, _image: Image) -> Image:
        """
        Process a PIL image

        :param _image: PIL image
        :return: processed PIL image
        """

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.width * _image.height))
        out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.RealESRGANImage(in_bytes, _image.width, _image.height, channels)

        self.raw_out_image = wrapped.RealESRGANImage(
            out_bytes,
            self._scale * _image.width,
            self._scale * _image.height,
            channels,
        )

        self.process()

        return Image.frombytes(
            _image.mode,
            (
                self._scale * _image.width,
                self._scale * _image.height,
            ),
            self.raw_out_image.get_data(),
        )

    def process_cv2(self, _image: np.ndarray) -> np.ndarray:
        """
        Process a cv2 image

        :param _image: cv2 image
        :return: processed cv2 image
        """
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.shape[1] * _image.shape[0]))
        out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.RealESRGANImage(in_bytes, _image.shape[1], _image.shape[0], channels)

        self.raw_out_image = wrapped.RealESRGANImage(
            out_bytes,
            self._scale * _image.shape[1],
            self._scale * _image.shape[0],
            channels,
        )

        self.process()

        res = np.frombuffer(self.raw_out_image.get_data(), dtype=np.uint8).reshape(
            self._scale * _image.shape[0], self._scale * _image.shape[1], channels
        )

        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    def process_bytes(self, _image_bytes: bytes, width: int, height: int, channels: int) -> bytes:
        """
        Process a bytes image, like bytes from ffmpeg

        :param _image_bytes: bytes
        :param width: image width
        :param height: image height
        :param channels: image channels
        :return: processed bytes image
        """
        if self.raw_in_image is None and self.raw_out_image is None:
            self.raw_in_image = wrapped.RealESRGANImage(_image_bytes, width, height, channels)

            self.raw_out_image = wrapped.RealESRGANImage(
                (self._scale**2) * len(_image_bytes) * b"\x00",
                self._scale * width,
                self._scale * height,
                channels,
            )

        self.raw_in_image.set_data(_image_bytes)

        self.process()

        return self.raw_out_image.get_data()
