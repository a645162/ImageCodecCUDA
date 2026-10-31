/*
 * File: tiff_container.cpp
 * Brief: Tiff container.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tiff_container.cpp - TIFF 6.0 容器读写实现（内存语义）。
#include "tiff_container.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace image_codec {
namespace cuda {
namespace tiff {

// ---------------------------------------------------------------------------
// TiffReader
// ---------------------------------------------------------------------------
bool TiffReader::openMemory(const void *data, size_t size) {
  close();
  if (data == nullptr || size < 8) return false;
  const uint8_t *p = static_cast<const uint8_t *>(data);
  file_.assign(p, p + size);
  return parseHeader() && parseIfd();
}

void TiffReader::close() {
  file_.clear();
  entries_.clear();
  tag_index_.clear();
  compression_ = kCompressionNone;
  valid_ = false;
}

bool TiffReader::parseHeader() {
  if (file_.size() < 8) return false;
  if (file_[0] == 'I' && file_[1] == 'I') {
    order_ = ByteOrder::Little;
  } else if (file_[0] == 'M' && file_[1] == 'M') {
    order_ = ByteOrder::Big;
  } else {
    return false;
  }
  magic_ = readU16(2);
  if (magic_ != 42) return false;
  ifd_offset_ = readU32(4);
  if (static_cast<uint64_t>(ifd_offset_) + 2 > file_.size()) return false;
  valid_ = true;
  return true;
}

bool TiffReader::parseIfd() {
  if (!valid_) return false;
  ifd_count_ = readU16(ifd_offset_);
  const size_t base = static_cast<size_t>(ifd_offset_) + 2;
  if (base + static_cast<size_t>(ifd_count_) * 12 + 4 > file_.size()) {
    return false;
  }
  entries_.reserve(ifd_count_);
  for (uint16_t i = 0; i < ifd_count_; ++i) {
    const size_t eo = base + static_cast<size_t>(i) * 12;
    IfdEntry e;
    e.tag = readU16(eo);
    e.type = readU16(eo + 2);
    e.count = readU32(eo + 4);
    std::memcpy(e.value_bytes, file_.data() + eo + 8, 4);
    e.value = readU32Bytes(e.value_bytes);
    entries_.push_back(e);
    tag_index_[e.tag] = i;
  }
  uint16_t comp = kCompressionNone;
  if (getU16(kTagCompression, comp)) compression_ = comp;
  return true;
}

uint16_t TiffReader::readU16(size_t off) const {
  if (off + 2 > file_.size()) return 0;
  return readU16Bytes(file_.data() + off);
}

uint32_t TiffReader::readU32(size_t off) const {
  if (off + 4 > file_.size()) return 0;
  return readU32Bytes(file_.data() + off);
}

uint16_t TiffReader::readU16Bytes(const uint8_t *p) const {
  if (order_ == ByteOrder::Little) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
  }
  return static_cast<uint16_t>((p[0] << 8) | p[1]);
}

uint32_t TiffReader::readU32Bytes(const uint8_t *p) const {
  if (order_ == ByteOrder::Little) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
  }
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

bool TiffReader::readEntryRaw(const IfdEntry &e,
                              std::vector<uint8_t> &out) const {
  const size_t ts = tiffTypeSize(e.type);
  if (ts == 0) return false;
  const size_t nbytes = ts * e.count;
  out.clear();
  if (nbytes <= 4) {
    // 值内联在 Value 字段前 nbytes 字节（按文件字节序）。
    out.assign(e.value_bytes, e.value_bytes + nbytes);
    return true;
  }
  if (static_cast<uint64_t>(e.value) + nbytes > file_.size()) return false;
  out.assign(file_.begin() + e.value, file_.begin() + e.value + nbytes);
  return true;
}

bool TiffReader::getU16(uint16_t tag, uint16_t &v) const {
  auto it = tag_index_.find(tag);
  if (it == tag_index_.end()) return false;
  const IfdEntry &e = entries_[it->second];
  if (e.count == 0) return false;
  std::vector<uint8_t> raw;
  if (!readEntryRaw(e, raw)) return false;
  if (e.type == kTypeShort) {
    v = readU16Bytes(raw.data());
    return true;
  }
  if (e.type == kTypeLong) {
    v = static_cast<uint16_t>(readU32Bytes(raw.data()));
    return true;
  }
  if (e.type == kTypeByte) {
    v = raw[0];
    return true;
  }
  return false;
}

bool TiffReader::getU32(uint16_t tag, uint32_t &v) const {
  auto it = tag_index_.find(tag);
  if (it == tag_index_.end()) return false;
  const IfdEntry &e = entries_[it->second];
  if (e.count == 0) return false;
  std::vector<uint8_t> raw;
  if (!readEntryRaw(e, raw)) return false;
  if (e.type == kTypeLong) {
    v = readU32Bytes(raw.data());
    return true;
  }
  if (e.type == kTypeShort) {
    v = readU16Bytes(raw.data());
    return true;
  }
  return false;
}

bool TiffReader::getU32Array(uint16_t tag, std::vector<uint32_t> &v) const {
  auto it = tag_index_.find(tag);
  if (it == tag_index_.end()) return false;
  const IfdEntry &e = entries_[it->second];
  v.clear();
  if (e.count == 0) return true;
  std::vector<uint8_t> raw;
  if (!readEntryRaw(e, raw)) return false;
  v.reserve(e.count);
  if (e.type == kTypeShort) {
    for (uint32_t i = 0; i < e.count; ++i) {
      v.push_back(readU16Bytes(raw.data() + i * 2));
    }
    return true;
  }
  if (e.type == kTypeLong) {
    for (uint32_t i = 0; i < e.count; ++i) {
      v.push_back(readU32Bytes(raw.data() + i * 4));
    }
    return true;
  }
  return false;
}

bool TiffReader::imageInfo(int &w, int &h, int &spp, int &bps, int &photometric,
                           int &compression, int &planar) const {
  uint32_t u32 = 0;
  uint16_t u16 = 0;
  if (!getU32(kTagImageWidth, u32)) return false;
  w = static_cast<int>(u32);
  if (!getU32(kTagImageLength, u32)) return false;
  h = static_cast<int>(u32);
  // 可选 tag 按 TIFF 默认值补齐。
  spp = getU16(kTagSamplesPerPixel, u16) ? u16 : 1;
  bps = getU16(kTagBitsPerSample, u16) ? u16 : 1;
  if (!getU16(kTagPhotometric, u16)) return false;
  photometric = u16;
  compression = getU16(kTagCompression, u16) ? u16 : kCompressionNone;
  planar = getU16(kTagPlanarConfig, u16) ? u16 : 1;
  return true;
}

bool TiffReader::getStripInfo(std::vector<uint32_t> &offsets,
                              std::vector<uint32_t> &counts,
                              uint32_t &rows_per_strip) const {
  if (!getU32Array(kTagStripOffsets, offsets)) return false;
  if (!getU32Array(kTagStripByteCounts, counts)) return false;
  if (offsets.size() != counts.size() || offsets.empty()) return false;
  if (!getU32(kTagRowsPerStrip, rows_per_strip)) rows_per_strip = 0;
  return true;
}

bool TiffReader::getTileInfo(std::vector<uint32_t> &offsets,
                             std::vector<uint32_t> &counts, uint32_t &tile_w,
                             uint32_t &tile_h) const {
  if (!getU32Array(kTagTileOffsets, offsets)) return false;
  if (!getU32Array(kTagTileByteCounts, counts)) return false;
  if (offsets.size() != counts.size() || offsets.empty()) return false;
  if (!getU32(kTagTileWidth, tile_w)) return false;
  if (!getU32(kTagTileLength, tile_h)) return false;
  return true;
}

bool TiffReader::loadBytes(uint32_t offset, uint32_t length,
                           std::vector<uint8_t> &out) const {
  out.clear();
  if (static_cast<uint64_t>(offset) + length > file_.size()) return false;
  out.assign(file_.begin() + offset, file_.begin() + offset + length);
  return true;
}

// ---------------------------------------------------------------------------
// TiffWriter
// ---------------------------------------------------------------------------
void TiffWriter::reset() {
  tiled_ = false;
  w_ = h_ = 0;
  rows_per_strip_ = 1;
  tile_w_ = tile_h_ = 0;
  compression_ = kCompressionNone;
  n_ = 0;
  offs_.clear();
  sizes_.clear();
  data_.clear();
  buf_.clear();
}

bool TiffWriter::openStrip(int w, int h, int channels, int rows_per_strip,
                           uint16_t compression, size_t n_strips) {
  reset();
  if (w <= 0 || h <= 0 || rows_per_strip <= 0 || n_strips == 0) return false;
  if (channels < 1 || channels > 4) return false;
  tiled_ = false;
  w_ = w;
  h_ = h;
  channels_ = channels;
  rows_per_strip_ = rows_per_strip;
  compression_ = compression;
  n_ = n_strips;
  offs_.assign(n_, 0);
  sizes_.assign(n_, 0);
  return true;
}

bool TiffWriter::openTile(int w, int h, int channels, uint32_t tile_w,
                          uint32_t tile_h, uint16_t compression,
                          size_t n_tiles) {
  reset();
  if (w <= 0 || h <= 0 || tile_w == 0 || tile_h == 0 || n_tiles == 0) {
    return false;
  }
  if (channels < 1 || channels > 4) return false;
  tiled_ = true;
  w_ = w;
  h_ = h;
  channels_ = channels;
  tile_w_ = tile_w;
  tile_h_ = tile_h;
  compression_ = compression;
  n_ = n_tiles;
  offs_.assign(n_, 0);
  sizes_.assign(n_, 0);
  return true;
}

bool TiffWriter::appendStrip(size_t idx, const void *data, size_t size) {
  if (tiled_ || idx >= n_ || data == nullptr) return false;
  offs_[idx] = static_cast<uint32_t>(data_.size());
  sizes_[idx] = static_cast<uint32_t>(size);
  const uint8_t *p = static_cast<const uint8_t *>(data);
  data_.insert(data_.end(), p, p + size);
  return true;
}

bool TiffWriter::appendTile(size_t idx, const void *data, size_t size) {
  if (!tiled_ || idx >= n_ || data == nullptr) return false;
  offs_[idx] = static_cast<uint32_t>(data_.size());
  sizes_[idx] = static_cast<uint32_t>(size);
  const uint8_t *p = static_cast<const uint8_t *>(data);
  data_.insert(data_.end(), p, p + size);
  return true;
}

void TiffWriter::putU16(size_t at, uint16_t v) {
  buf_[at] = static_cast<uint8_t>(v & 0xFF);
  buf_[at + 1] = static_cast<uint8_t>(v >> 8);
}

void TiffWriter::putU32(size_t at, uint32_t v) {
  buf_[at] = static_cast<uint8_t>(v & 0xFF);
  buf_[at + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  buf_[at + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  buf_[at + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

void TiffWriter::build() {
  const int spp = channels_;
  // 1/2 -> BlackIsZero（灰度、灰度+A），3/4 -> RGB（RGB、RGBA）。
  const uint16_t photometric = spp >= 3 ? 2 : 1;
  // 多出来的那个样本是 Alpha：spp=2 与 spp=4 都必须写 ExtraSamples，
  // 否则 spp=4 会被解码器按 CMYK 解释，颜色整体错位。
  const bool has_alpha = (spp == 2 || spp == 4);

  const size_t n_tags = (tiled_ ? 11 : 10) + (has_alpha ? 1 : 0);
  const size_t ifd_size = 2 + n_tags * 12 + 4;
  const size_t off_bps = 8 + ifd_size;
  // 只有数组总字节数 > 4（spp >= 3）时才需要把 BitsPerSample 放到 Value 之外；
  // spp==2 的两个 SHORT 恰好 4 字节，必须内联成 0x00080008，否则读回来是
  // 「位深 = 数组偏移」这种离谱数字。
  const bool bps_array = spp >= 3;
  const size_t bps_bytes = bps_array ? static_cast<size_t>(spp) * 2 : 0;
  const size_t off_off = off_bps + bps_bytes;
  const size_t off_cnt = off_off + n_ * 4;
  const size_t data_start = off_cnt + n_ * 4;

  buf_.assign(data_start + data_.size(), 0);
  buf_[0] = 'I';
  buf_[1] = 'I';
  buf_[2] = 42;
  buf_[3] = 0;
  putU32(4, 8); // IFD offset

  size_t e = 8;
  putU16(e, static_cast<uint16_t>(n_tags));
  e += 2;

  // 先攒下来再排序写出：TIFF 规范要求 IFD 条目按 tag 升序，libtiff 对非升序
  // 会打 "tags are not sorted in ascending order" 警告。之前 strip 布局把
  // 273/278/279 写在 284 之后（tiled 恰好是升序，所以只有 strip 出问题）——
  // 主流解码器仍能读，但一个自己写出来的文件不该带着规范违背。
  struct IfdEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    uint32_t value = 0;
  };
  std::vector<IfdEntry> entries;
  entries.reserve(n_tags + 1);

  auto entry = [&entries](uint16_t tag, uint16_t type, uint32_t count,
                          uint32_t val) {
    entries.push_back(IfdEntry{tag, type, count, val});
  };

  // count==1 的 LONG/SHORT 必须把值直接写进 Value 字段；只有 count>1 时
  // Value 才存数组偏移（这一条曾让单 strip 的 TIFF 全部损坏）。
  const uint32_t strip_offsets_val =
      static_cast<uint32_t>(n_ == 1 ? data_start + offs_[0] : off_off);
  const uint32_t strip_counts_val =
      static_cast<uint32_t>(n_ == 1 ? sizes_[0] : off_cnt);

  entry(kTagImageWidth, kTypeLong, 1, static_cast<uint32_t>(w_));
  entry(kTagImageLength, kTypeLong, 1, static_cast<uint32_t>(h_));
  entry(kTagBitsPerSample, kTypeShort, static_cast<uint32_t>(spp),
        spp == 1 ? 8u
                 : (spp == 2 ? 0x00080008u : static_cast<uint32_t>(off_bps)));
  entry(kTagCompression, kTypeShort, 1, compression_);
  entry(kTagPhotometric, kTypeShort, 1, photometric);
  entry(kTagSamplesPerPixel, kTypeShort, 1, static_cast<uint32_t>(spp));
  entry(kTagPlanarConfig, kTypeShort, 1, 1); // chunky
  if (has_alpha) {
    entry(338 /* ExtraSamples */, kTypeShort, 1, 2 /* unassociated alpha */);
  }
  if (!tiled_) {
    entry(kTagStripOffsets, kTypeLong, static_cast<uint32_t>(n_),
          strip_offsets_val);
    entry(kTagRowsPerStrip, kTypeLong, 1,
          static_cast<uint32_t>(rows_per_strip_));
    entry(kTagStripByteCounts, kTypeLong, static_cast<uint32_t>(n_),
          strip_counts_val);
  } else {
    entry(kTagTileWidth, kTypeLong, 1, tile_w_);
    entry(kTagTileLength, kTypeLong, 1, tile_h_);
    entry(kTagTileOffsets, kTypeLong, static_cast<uint32_t>(n_),
          strip_offsets_val);
    entry(kTagTileByteCounts, kTypeLong, static_cast<uint32_t>(n_),
          strip_counts_val);
  }
  std::sort(entries.begin(), entries.end(),
            [](const IfdEntry &a, const IfdEntry &b) { return a.tag < b.tag; });
  for (const IfdEntry &item : entries) {
    putU16(e, item.tag);
    putU16(e + 2, item.type);
    putU32(e + 4, item.count);
    putU32(e + 8, item.value);
    e += 12;
  }

  putU32(e, 0); // next IFD

  // 只有真的把 BitsPerSample 放到数组区时才写这一段：spp<=2 时 off_bps 与
  // strip 偏移数组重合，无条件写会覆盖掉第一个 strip 的偏移（单 strip 恰好
  // 走得通，多 strip 直接坏）。
  if (bps_array) {
    for (int k = 0; k < spp; ++k) {
      putU16(off_bps + static_cast<size_t>(k) * 2, 8);
    }
  }
  for (size_t i = 0; i < n_; ++i) {
    putU32(off_off + i * 4, static_cast<uint32_t>(data_start + offs_[i]));
  }
  for (size_t i = 0; i < n_; ++i) {
    putU32(off_cnt + i * 4, sizes_[i]);
  }
  if (!data_.empty()) {
    std::memcpy(buf_.data() + data_start, data_.data(), data_.size());
  }
}

const std::vector<uint8_t> &TiffWriter::buffer() {
  if (buf_.empty()) {
    build();
  }
  return buf_;
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
