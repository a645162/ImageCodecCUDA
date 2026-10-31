/*
 * File: tiff_container.h
 * Brief: Tiff container.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tiff_container.h - TIFF 6.0 容器读写（LE/BE 读取、LE 写出、strip/tile 布局）。
//
// 只做容器：压缩/解压由调用方提供字节流。宿主侧实现，与 GPU 无关。
// 读：openMemory() 后按 tag 取值；写：openStrip/openTile -> append* -> buffer()。
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace image_codec {
namespace cuda {
namespace tiff {

// ---- TIFF 6.0 常用 Tag/Type/值常量 ----
enum TiffTag : uint16_t {
  kTagImageWidth = 256,
  kTagImageLength = 257,
  kTagBitsPerSample = 258,
  kTagCompression = 259,
  kTagPhotometric = 262,
  kTagStripOffsets = 273,
  kTagSamplesPerPixel = 277,
  kTagRowsPerStrip = 278,
  kTagStripByteCounts = 279,
  kTagPlanarConfig = 284,
  kTagTileWidth = 322,
  kTagTileLength = 323,
  kTagTileOffsets = 324,
  kTagTileByteCounts = 325,
};

enum TiffType : uint16_t {
  kTypeByte = 1,
  kTypeAscii = 2,
  kTypeShort = 3,
  kTypeLong = 4,
  kTypeRational = 5,
};

enum TiffCompressionValue : uint16_t {
  kCompressionNone = 1,
  kCompressionLzw = 5,
};

// TIFF 字段类型字节数；未知类型返回 0。
inline size_t tiffTypeSize(uint16_t type) {
  switch (type) {
  case kTypeByte:
    return 1;
  case kTypeAscii:
    return 1;
  case kTypeShort:
    return 2;
  case kTypeLong:
    return 4;
  case kTypeRational:
    return 8;
  default:
    return 0;
  }
}

// ---------------------------------------------------------------------------
// TiffReader：解析 header + 第一个 IFD，提供 typed 读取与 strip/tile 数据访问。
// ---------------------------------------------------------------------------
class TiffReader {
public:
  bool openMemory(const void *data, size_t size);
  void close();

  bool valid() const { return valid_; }
  bool isTiled() const {
    return hasTag(kTagTileWidth) && hasTag(kTagTileOffsets);
  }
  uint16_t compression() const { return compression_; }

  bool hasTag(uint16_t tag) const { return tag_index_.count(tag) != 0; }
  bool getU16(uint16_t tag, uint16_t &v) const;
  bool getU32(uint16_t tag, uint32_t &v) const;
  bool getU32Array(uint16_t tag, std::vector<uint32_t> &v) const;

  // 图像元数据（bps=每样本位数；spp=样本/像素；photometric/compression/planar）。
  bool imageInfo(int &w, int &h, int &spp, int &bps, int &photometric,
                 int &compression, int &planar) const;

  bool getStripInfo(std::vector<uint32_t> &offsets,
                    std::vector<uint32_t> &counts,
                    uint32_t &rows_per_strip) const;
  bool getTileInfo(std::vector<uint32_t> &offsets,
                   std::vector<uint32_t> &counts, uint32_t &tile_w,
                   uint32_t &tile_h) const;

  // 读取 strip/tile 的原始（压缩）字节。
  bool loadBytes(uint32_t offset, uint32_t length,
                 std::vector<uint8_t> &out) const;

private:
  enum class ByteOrder { Little, Big };

  struct IfdEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    uint8_t value_bytes[4] = {0, 0, 0, 0}; // Value 字段原始 4 字节
    uint32_t value = 0;                    // value_bytes 按文件字节序解释
  };

  bool parseHeader();
  bool parseIfd();
  bool readEntryRaw(const IfdEntry &e, std::vector<uint8_t> &out) const;
  uint16_t readU16(size_t off) const;
  uint32_t readU32(size_t off) const;
  uint16_t readU16Bytes(const uint8_t *p) const;
  uint32_t readU32Bytes(const uint8_t *p) const;

  std::vector<uint8_t> file_;
  ByteOrder order_ = ByteOrder::Little;
  uint16_t magic_ = 0;
  uint32_t ifd_offset_ = 0;
  uint16_t ifd_count_ = 0;
  std::vector<IfdEntry> entries_;
  std::map<uint16_t, size_t> tag_index_;
  uint16_t compression_ = kCompressionNone;
  bool valid_ = false;
};

// ---------------------------------------------------------------------------
// TiffWriter：strip 或 tile 布局 TIFF 写出（little-endian）。压缩数据由调用方提供。
// 用法：openStrip(...) -> appendStrip(...) * n -> buffer()。
// ---------------------------------------------------------------------------
class TiffWriter {
public:
  void reset();

  // channels: 1=灰度(BlackIsZero), 3=RGB, 4=RGBA(ExtraSamples=unassociated)。
  bool openStrip(int w, int h, int channels, int rows_per_strip,
                 uint16_t compression, size_t n_strips);
  bool openTile(int w, int h, int channels, uint32_t tile_w, uint32_t tile_h,
                uint16_t compression, size_t n_tiles);

  bool appendStrip(size_t idx, const void *data, size_t size);
  bool appendTile(size_t idx, const void *data, size_t size);

  // 组装并返回完整 TIFF 字节流（首次调用后缓存）。
  const std::vector<uint8_t> &buffer();

private:
  void build();
  void putU16(size_t at, uint16_t v);
  void putU32(size_t at, uint32_t v);

  bool tiled_ = false;
  int w_ = 0;
  int h_ = 0;
  int channels_ = 3;
  int rows_per_strip_ = 1;
  uint32_t tile_w_ = 0;
  uint32_t tile_h_ = 0;
  uint16_t compression_ = kCompressionNone;
  size_t n_ = 0;
  std::vector<uint32_t> offs_;  // 每 strip/tile 在 payload 内的偏移
  std::vector<uint32_t> sizes_; // 每 strip/tile 的字节数
  std::vector<uint8_t> data_;   // payload 区（追加）
  std::vector<uint8_t> buf_;    // 最终组装结果
};

} // namespace tiff
} // namespace cuda
} // namespace image_codec
