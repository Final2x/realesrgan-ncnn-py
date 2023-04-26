#include "realesrgan_wrapped.h"

// Image Data Structure
RealESRGANImage::RealESRGANImage(std::string d, int w, int h, int c) {
    this->d = std::move(d);
    this->w = w;
    this->h = h;
    this->c = c;
}

void RealESRGANImage::set_data(std::string data) {
    this->d = std::move(data);
}

pybind11::bytes RealESRGANImage::get_data() const {
    return pybind11::bytes(this->d);
}

// RealESRGANWrapped
RealESRGANWrapped::RealESRGANWrapped(int gpuid, bool tta_mode): RealESRGAN(gpuid, tta_mode) {
    this->gpuid = gpuid;
}

int RealESRGANWrapped::get_tilesize() const {
    int tilesize = 0;

    if (this->gpuid < 0) {
        return 400;
    }

    uint32_t heap_budget = ncnn::get_gpu_device(this->gpuid)->get_heap_budget();

    if (heap_budget >= 1900) {
        tilesize = 200;
    } else if (heap_budget >= 550) {
        tilesize = 100;
    } else if (heap_budget >= 190) {
        tilesize = 64;
    } else {
        tilesize = 32;
    }

    return tilesize;
}

void RealESRGANWrapped::set_parameters(int _tilesize, int _scale) {
    RealESRGAN::tilesize = _tilesize ? _tilesize : RealESRGANWrapped::get_tilesize();
    RealESRGAN::scale = _scale;
    RealESRGAN::prepadding = 10;
}

int RealESRGANWrapped::load(const std::string &parampath, const std::string &modelpath) {
#if _WIN32
    // convert string to wstring
    auto to_wide_string = [&](const std::string& input) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(input);
    };
    return RealESRGAN::load(to_wide_string(parampath), to_wide_string(modelpath));
#else
    return RealESRGAN::load(parampath, modelpath);
#endif
}

int RealESRGANWrapped::process(const RealESRGANImage &inimage, RealESRGANImage &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return RealESRGAN::process(inimagemat, outimagemat);
}

int get_gpu_count() { return ncnn::get_gpu_count(); }

void destroy_gpu_instance() { ncnn::destroy_gpu_instance(); }

PYBIND11_MODULE(realesrgan_ncnn_vulkan_wrapper, m) {
    pybind11::class_<RealESRGANWrapped>(m, "RealESRGANWrapped")
            .def(pybind11::init<int, bool>())
            .def("load", &RealESRGANWrapped::load)
            .def("process", &RealESRGANWrapped::process)
            .def("set_parameters", &RealESRGANWrapped::set_parameters);

    pybind11::class_<RealESRGANImage>(m, "RealESRGANImage")
            .def(pybind11::init<std::string, int, int, int>())
            .def("get_data", &Image::get_data)
            .def("set_data", &Image::set_data);

    m.def("get_gpu_count", &get_gpu_count);

    m.def("destroy_gpu_instance", &destroy_gpu_instance);
}
