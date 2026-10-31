/*
 * File: lzw_codec.h
 * Brief: Lzw codec.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_codec.h - TIFF LZW 的 CPU 参考实现（编码 + 解码）。
//
// 语义与 libtiff 对齐：
//   * 码宽 9->10->11->12 于 next_code == 512/1024/2048 切换；
//   * 字典满(next_code == 4096) 后停止建表，收到 ClearCode(256) 重置，EOI(257) 结束；
//   * 解码器建表滞后编码器一位，故宽度切换点在 511/1023/2047。
// GPU kernel 与本实现必须逐字节一致（tests 里有对拍）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace image_codec {
namespace cuda {
namespace lzw {

// 压缩一段字节流（每段独立，段首发 ClearCode）。
int encode(const uint8_t *in, size_t in_size, std::vector<uint8_t> &out);

// 解压一段字节流。输入不足/非法码返回负值。
int decode(const uint8_t *in, size_t in_size, std::vector<uint8_t> &out);

} // namespace lzw
} // namespace cuda
} // namespace image_codec
