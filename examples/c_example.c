// c_example.c - 纯 C 调用示例：TIFF -> PNG 转码（全部走内存）。
//
// 构建：cmake --build build --target icu_c_example
// 运行：icu_c_example in.tiff out.png
#include <ImageCodecCUDA/image_codec_cuda.h>

#include <stdio.h>
#include <stdlib.h>

static unsigned char *read_all(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  const long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len <= 0) {
    fclose(f);
    return NULL;
  }
  unsigned char *buf = (unsigned char *)malloc((size_t)len);
  if (buf == NULL) {
    fclose(f);
    return NULL;
  }
  const size_t got = fread(buf, 1, (size_t)len, f);
  fclose(f);
  if (got != (size_t)len) {
    free(buf);
    return NULL;
  }
  *out_size = (size_t)len;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: icu_c_example <in> <out.png>\n");
    return 1;
  }
  size_t in_size = 0;
  unsigned char *in = read_all(argv[1], &in_size);
  if (in == NULL) {
    fprintf(stderr, "read failed: %s\n", argv[1]);
    return 2;
  }

  image_codec_cuda_codec *codec = NULL;
  image_codec_cuda_status st = image_codec_cuda_codec_create(&codec, 0);
  if (st != IMAGE_CODEC_CUDA_OK) {
    fprintf(stderr, "create codec failed: %s\n",
            image_codec_cuda_status_string(st));
    free(in);
    return 3;
  }

  image_codec_cuda_image img;
  st = image_codec_cuda_decode(codec, IMAGE_CODEC_CUDA_FORMAT_UNKNOWN, in,
                               in_size, &img);
  free(in);
  if (st != IMAGE_CODEC_CUDA_OK) {
    fprintf(stderr, "decode failed: %s\n", image_codec_cuda_status_string(st));
    image_codec_cuda_codec_destroy(codec);
    return 4;
  }
  printf("decoded %ux%u x%u\n", img.width, img.height, img.channels);

  image_codec_cuda_encode_params params = {0};
  params.png_compression = 6;
  params.png_filter = -1;

  size_t need = 0;
  st = image_codec_cuda_encode(codec, IMAGE_CODEC_CUDA_FORMAT_PNG, &img,
                               &params, NULL, &need, 0);
  if (st != IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY && st != IMAGE_CODEC_CUDA_OK) {
    fprintf(stderr, "encode(size) failed: %s\n",
            image_codec_cuda_status_string(st));
    image_codec_cuda_image_free(&img);
    image_codec_cuda_codec_destroy(codec);
    return 5;
  }
  unsigned char *enc = (unsigned char *)malloc(need);
  st = image_codec_cuda_encode(codec, IMAGE_CODEC_CUDA_FORMAT_PNG, &img,
                               &params, enc, &need, need);
  image_codec_cuda_image_free(&img);
  image_codec_cuda_codec_destroy(codec);
  if (st != IMAGE_CODEC_CUDA_OK) {
    fprintf(stderr, "encode failed: %s\n", image_codec_cuda_status_string(st));
    free(enc);
    return 6;
  }

  FILE *f = fopen(argv[2], "wb");
  if (f == NULL) {
    fprintf(stderr, "write failed: %s\n", argv[2]);
    free(enc);
    return 7;
  }
  fwrite(enc, 1, need, f);
  fclose(f);
  free(enc);
  printf("wrote %s (%zu bytes)\n", argv[2], need);
  return 0;
}
