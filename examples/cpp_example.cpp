// cpp_example.cpp - C++ RAII 示例：PNG -> JPEG 转码并打印设备能力。
#include <ImageCodecCUDA/image_codec_cuda.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
  try {
    for (const auto &name : image_codec_cuda::deviceList()) {
      std::printf("device: %s\n", name.c_str());
    }
    image_codec_cuda::Codec codec(0);
    const auto caps = codec.capabilities();
    std::printf(
        "caps: jpeg=%d/%d tiff=%d/%d png=%d/%d lzw_gpu=%d png_filter_gpu=%d\n",
        caps.jpeg_decode, caps.jpeg_encode, caps.tiff_decode, caps.tiff_encode,
        caps.png_decode, caps.png_encode, caps.tiff_lzw_gpu,
        caps.png_filter_gpu);
    if (caps.note[0] != '\0') {
      std::printf("note: %s\n", caps.note);
    }
    if (argc < 2) {
      std::puts("usage: icu_cpp_example <in.png> [out.jpg]");
      return 0;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "read failed: %s\n", argv[1]);
      return 2;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    const int fmt = image_codec_cuda::probeFormat(bytes.data(), bytes.size());
    const image_codec_cuda::Image img =
        codec.decode(fmt, bytes.data(), bytes.size());
    std::printf("decoded %ux%u x%u\n", img.width, img.height, img.channels);

    if (argc >= 3) {
      image_codec_cuda::EncodeParams params;
      params.jpeg_quality = 92;
      const auto jpg = codec.encode(IMAGE_CODEC_CUDA_FORMAT_JPEG, img, params);
      std::ofstream out(argv[2], std::ios::binary);
      out.write(reinterpret_cast<const char *>(jpg.data()),
                static_cast<std::streamsize>(jpg.size()));
      std::printf("wrote %s (%zu bytes)\n", argv[2], jpg.size());
    }
    return 0;
  } catch (const image_codec_cuda::Error &e) {
    std::fprintf(stderr, "error (%d): %s\n", static_cast<int>(e.status()),
                 e.what());
    return 1;
  }
}
