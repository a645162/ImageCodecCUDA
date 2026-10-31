/*
 * File: platform.h
 * Brief: Platform.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// platform.h - 与 GPU 无关的宿主侧小工具（计时、休眠）。
//
// 与 ImageCodecPlatform 的 image_codec_* 平台抽象刻意重名区隔：本库是被 add_subdirectory
// 复用的后端库，不能链接另一个子模块的 C 符号，否则同一进程内会出现两份实现。
#pragma once

namespace image_codec {
namespace cuda {

// 单调时钟纳秒（不受系统时间调整影响）。
long long nowNs();

// 休眠（毫秒）。
void sleepMs(unsigned ms);

} // namespace cuda
} // namespace image_codec
