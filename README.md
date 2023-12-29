# realesrgan-ncnn-py

Python Binding for realesrgan-ncnn-py with PyBind11

[![PyPI version](https://badge.fury.io/py/realesrgan-ncnn-py.svg?123456)](https://badge.fury.io/py/realesrgan-ncnn-py?123456)
[![Release](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/Release.yml/badge.svg)](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/Release.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/realesrgan-ncnn-py)

Real-ESRGAN aims at developing Practical Algorithms for General Image/Video Restoration.
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure
synthetic data.
This wrapper provides an easy-to-use interface for running the pre-trained Real-ESRGAN model.

### Current building status matrix

|    System     |                                                                                                               Status                                                                                                                | CPU (32bit) |    CPU (64bit)     | GPU (32bit) |    GPU (64bit)     |
| :-----------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------: | :----------------: | :---------: | :----------------: |
| Linux (Clang) |         [![CI-Linux-x64-Clang](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml/badge.svg)](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml)         |      —      | :white_check_mark: |      —      | :white_check_mark: |
|  Linux (GCC)  |            [![CI-Linux-x64-GCC](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml/badge.svg)](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml)            |      —      | :white_check_mark: |      —      | :white_check_mark: |
|    Windows    |       [![CI-Windows-x64-MSVC](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml/badge.svg)](https://github.com/Tohrusky/realesrgan-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml)        |      —      | :white_check_mark: |      —      | :white_check_mark: |
|     MacOS     | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) |      —      | :white_check_mark: |      —      | :white_check_mark: |
|  MacOS (ARM)  | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/realcugan-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) |      —      | :white_check_mark: |      —      | :white_check_mark: |

# Usage

`Python >= 3.6 (>= 3.9 in MacOS arm)`

To use this package, simply install it via pip:

```sh
pip install realesrgan-ncnn-py
```

For Linux user:

```sh
apt install -y libomp5 libvulkan-dev
```

Then, import the Realesrgan class from the package:

```python
from realesrgan_ncnn_py import Realesrgan
```

To initialize the model:

```python
realesrgan = Realesrgan(gpuid: int = 0, tta_mode: bool = False, tilesize: int = 0, model: int = 0)
# model can be -1, 0, 1, 2, 3, 4; 0 for default, -1 for custom load
# 0: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2},
# 1: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3},
# 2: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4},
# 3: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4},
# 4: {"param": "realesrgan-x4plus.param", "bin": "realesrgan-x4plus.bin", "scale": 4}


```

Here, gpuid specifies the GPU device to use, tta_mode enables test-time augmentation, tilesize specifies the tile size
for processing (0 or >= 32), and model specifies the num of the pre-trained model to use.

Once the model is initialized, you can use the upscale method to super-resolve your images:

### Pillow

```python
from PIL import Image

realesrgan = Realesrgan(gpuid=0)
with Image.open("input.jpg") as image:
    image = realesrgan.process_pil(image)
    image.save("output.jpg", quality=95)
```

### opencv-python

```python
import cv2

realesrgan = Realesrgan(gpuid=0)
image = cv2.imdecode(np.fromfile("input.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
image = realesrgan.process_cv2(image)
cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")
```

### ffmpeg

```python
import subprocess as sp

# your ffmpeg parameters
command_out = [FFMPEG_BIN, ........]
command_in = [FFMPEG_BIN, ........]
pipe_out = sp.Popen(command_out, stdout=sp.PIPE, bufsize=10 ** 8)
pipe_in = sp.Popen(command_in, stdin=sp.PIPE)
realesrgan = Realesrgan(gpuid=0)
while True:
    raw_image = pipe_out.stdout.read(src_width * src_height * 3)
    if not raw_image:
        break
    raw_image = realesrgan.process_bytes(raw_image, src_width, src_height, 3)
    pipe_in.stdin.write(raw_image)
```

# Build

[here](https://github.com/Tohrusky/realesrgan-ncnn-py/blob/main/.github/workflows/Release.yml)

_The project just only been tested in Ubuntu 18+ and Debian 9+ environments on Linux, so if the project does not work on
your system, please try building it._

# References

The following references were used in the development of this project:

[xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) - This project was the main
inspiration for our work. It provided the core implementation of the Real-ESRGAN algorithm using the ncnn and Vulkan
libraries.

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Real-ESRGAN is an AI super resolution model, aims at developing
Practical Algorithms for General Image/Video Restoration.

[media2x/realsr-ncnn-vulkan-python](https://github.com/media2x/realsr-ncnn-vulkan-python) - This project was used as a
reference for implementing the wrapper. _Special thanks_ to the original author for sharing the code.

[ncnn](https://github.com/Tencent/ncnn) - ncnn is a high-performance neural network inference framework developed by
Tencent AI Lab.

# License

This project is licensed under the BSD 3-Clause - see
the [LICENSE file](https://github.com/Tohrusky/realesrgan-ncnn-py/blob/main/LICENSE) for details.
