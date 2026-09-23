#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_schedule.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Focused WinGen CSim: counted backing-memory adapters are TB-only. Production
// memory.cpp is exercised by the dual-model top CSim, not replaced there.
static std::vector<std::uint8_t> memory[2];
static unsigned aligned_reads = 0;
static unsigned failures = 0;

namespace esp_int8 {
void scheduled_window_generator_row(const tensor_desc_t&, const window_sched_desc_t&,
    hls::stream<act_vec_t>&, hls::stream<act_vec_t>&, u16_t, volatile u8_t&);
void csim_reset_narrow_transfer_words();
unsigned long long csim_get_narrow_transfer_words();

bool on_chip_memory_prepare_row_base(const tensor_desc_t& desc, i32_t h,
                                    u32_t& base, bool& valid) {
    valid = h >= 0 && h < static_cast<i32_t>(desc.h);
    base = 0;
    if (valid) {
        const unsigned channels = desc.reserved0.to_uint() ? desc.reserved0.to_uint() : desc.c.to_uint();
        base = desc.base_offset.to_uint() + h.to_uint() * desc.w.to_uint() * channels + desc.reserved1.to_uint();
    }
    return true;
}

bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t& desc,
                                             u32_t offset, act_vec_t& word) {
    word = 0;
    const unsigned bank = desc.bank_id.to_uint(), byte = offset.to_uint();
    if (bank >= 2 || (byte & 31U) || byte + 32 > memory[bank].size()) {
        ++failures;
        return false;
    }
    ++aligned_reads;
    for (int lane = 0; lane < 32; ++lane) {
        word.range(lane * 8 + 7, lane * 8) = memory[bank][byte + lane];
    }
    return true;
}

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc, i32_t h, i32_t w,
                                     u16_t c, u8_t valid_c, act_vec_t& word) {
    word = 0;
    if (h < 0 || w < 0 || h >= static_cast<i32_t>(desc.h) || w >= static_cast<i32_t>(desc.w)) {
        return true;
    }
    const unsigned channels = desc.reserved0.to_uint() ? desc.reserved0.to_uint() : desc.c.to_uint();
    const unsigned offset = desc.base_offset.to_uint() +
        (h.to_uint() * desc.w.to_uint() + w.to_uint()) * channels + desc.reserved1.to_uint() + c.to_uint();
    for (unsigned lane = 0; lane < valid_c.to_uint() && lane + c.to_uint() < desc.c.to_uint(); ++lane) {
        word.range(lane * 8 + 7, lane * 8) = memory[desc.bank_id.to_uint()][offset + lane];
    }
    return true;
}

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc, i32_t h,
                                          i32_t w, u16_t c, act_vec_t& word) {
    return on_chip_memory_read_packed_tile(desc, h, w, c, u8_t(32), word);
}
}

static unsigned u16(const std::vector<std::uint8_t>& data, unsigned offset) {
    return data.at(offset) | (unsigned(data.at(offset + 1)) << 8);
}

static unsigned u32(const std::vector<std::uint8_t>& data, unsigned offset) {
    return u16(data, offset) | (u16(data, offset + 2) << 16);
}

static esp_int8::window_sched_desc_t load_schedule(const std::vector<std::uint8_t>& data) {
    const unsigned conv = u32(data, 88);
    const unsigned off = u32(data, 92) + unsigned(data.at(conv + 2)) * 128;
    esp_int8::window_sched_desc_t s = {};
    s.mode=data.at(off); s.kernel=data.at(off+1); s.stride=data.at(off+2);
    s.dilation=data.at(off+3); s.padding=data.at(off+4); s.cache_chunks=data.at(off+5);
    s.cache_col_slots=data.at(off+6); s.flags=data.at(off+7);
    s.in_c=u16(data,off+8); s.out_w=u16(data,off+10); s.k_tiles=u16(data,off+12);
    s.cmd_base=u16(data,off+14); s.cmd_count=u16(data,off+16);
    for (int i=0; i<=esp_int8::MAX_K_TILE_COUNT; ++i) s.kt_cmd_base[i]=u16(data,off+18+i*2);
    s.loader_class=data.at(off+100); s.loader_request_cols=data.at(off+101);
    s.loader_warmup_issues=data.at(off+102); s.loader_warmup_new_cols=data.at(off+103);
    s.loader_steady_new_cols=data.at(off+104); s.loader_words_per_col=data.at(off+105);
    s.loader_warmup_mask=data.at(off+106); s.loader_steady_mask=data.at(off+107);
    for (int i=0; i<10; ++i) s.reserved[i]=u16(data,off+108+i*2);
    return s;
}

static esp_int8::window_sched_desc_t load_schedule_id(
    const std::vector<std::uint8_t>& data, unsigned schedule_id) {
    const unsigned off = u32(data, 92) + schedule_id * 128;
    esp_int8::window_sched_desc_t s = {};
    s.mode=data.at(off); s.kernel=data.at(off+1); s.stride=data.at(off+2);
    s.dilation=data.at(off+3); s.padding=data.at(off+4); s.cache_chunks=data.at(off+5);
    s.cache_col_slots=data.at(off+6); s.flags=data.at(off+7);
    s.in_c=u16(data,off+8); s.out_w=u16(data,off+10); s.k_tiles=u16(data,off+12);
    s.cmd_base=u16(data,off+14); s.cmd_count=u16(data,off+16);
    for (int i=0; i<=esp_int8::MAX_K_TILE_COUNT; ++i) s.kt_cmd_base[i]=u16(data,off+18+i*2);
    s.loader_class=data.at(off+100); s.loader_request_cols=data.at(off+101);
    s.loader_warmup_issues=data.at(off+102); s.loader_warmup_new_cols=data.at(off+103);
    s.loader_steady_new_cols=data.at(off+104); s.loader_words_per_col=data.at(off+105);
    s.loader_warmup_mask=data.at(off+106); s.loader_steady_mask=data.at(off+107);
    for (int i=0; i<10; ++i) s.reserved[i]=u16(data,off+108+i*2);
    return s;
}

static std::uint8_t expected_lane(const esp_int8::tensor_desc_t& src,
    int oh, int ow, int pad, int lane, int out_w) {
    if (lane >= 27 || ow >= out_w) return 0;
    const int spatial=lane/3, channel=lane%3;
    const int ih=oh*2-pad+spatial/3, iw=ow*2-pad+spatial%3;
    if (ih<0 || iw<0 || ih>=int(src.h.to_uint()) || iw>=int(src.w.to_uint())) return 0;
    return memory[src.bank_id.to_uint()][src.base_offset.to_uint() + (ih*src.w.to_uint()+iw)*3+channel];
}

static unsigned run_rows(esp_int8::window_sched_desc_t sched, const char* label,
    unsigned width, unsigned height, unsigned padding, bool reuse, unsigned bank,
    unsigned base, unsigned seed, unsigned first_row=0, unsigned count=0) {
    using namespace esp_int8;
    tensor_desc_t src = {};
    src.bank_id=bank; src.elem_bytes=1; src.base_offset=base;
    src.h=height; src.w=width; src.c=3;
    memory[bank].resize(base+height*width*3);
    for (unsigned i=0; i<height*width*3; ++i) memory[bank][base+i]=(i*17+seed*37+(i/width)*13)&255;
    const unsigned out_w=(width+2*padding-3)/2+1;
    const unsigned out_h=(height+2*padding-3)/2+1;
    sched.padding=padding; sched.out_w=out_w;
    sched.flags=1U | ((out_w&1U) ? 2U : 0U);
    sched.reserved[9]=reuse ? 2U | ((width*3/32)<<2) : 0U;
    aligned_reads=0;
    csim_reset_narrow_transfer_words();
    unsigned tokens=0;
    const unsigned stop=count ? first_row+count : out_h;
    for (unsigned oh=first_row; oh<stop; ++oh) {
        hls::stream<act_vec_t> act0, act1;
        u8_t state=0;
        scheduled_window_generator_row(src,sched,act0,act1,u16_t(oh),state);
        for (unsigned issue=0; issue<(out_w+1)/2; ++issue) {
            if (act0.empty() || act1.empty()) { ++failures; break; }
            const act_vec_t a=act0.read(), b=act1.read();
            ++tokens;
            for (int lane=0; lane<32; ++lane) {
                if (a.range(lane*8+7,lane*8).to_uint()!=expected_lane(src,oh,issue*2,padding,lane,out_w) ||
                    b.range(lane*8+7,lane*8).to_uint()!=expected_lane(src,oh,issue*2+1,padding,lane,out_w)) ++failures;
            }
        }
        if (!act0.empty() || !act1.empty() || state.to_uint()!=0U) ++failures;
    }
    if (tokens!=(stop-first_row)*((out_w+1)/2)) ++failures;
    const unsigned rows = stop - first_row;
    const unsigned issues = (out_w + 1) / 2;
    const unsigned long long expected_transfer = reuse
        ? static_cast<unsigned long long>(rows) * (15U + (issues - 1U) * 12U)
        : static_cast<unsigned long long>(rows) *
              ((issues - ((out_w & 1U) ? 1U : 0U)) * 18U +
               ((out_w & 1U) ? 9U : 0U));
    const unsigned long long transfer = csim_get_narrow_transfer_words();
    if (transfer != expected_transfer) ++failures;
    std::printf("[C3] %s words=%u source_reads=%u transfer=%llu expected=%llu act0=%u act1=%u errors=%u\n",
                label,width*3/32,aligned_reads,transfer,expected_transfer,tokens,tokens,failures);
    return aligned_reads;
}

static std::uint8_t expected_c12_lane(const esp_int8::tensor_desc_t& src,
    int oh, int ow, int lane, int kt) {
    const int linear_k = kt * 32 + lane;
    if (linear_k >= 9 * 12) return 0;
    const int spatial = linear_k / 12;
    const int channel = linear_k % 12;
    const int ih = oh - 1 + spatial / 3;
    const int iw = ow - 1 + spatial % 3;
    if (ih < 0 || iw < 0 || ih >= int(src.h.to_uint()) ||
        iw >= int(src.w.to_uint())) return 0;
    return memory[src.bank_id.to_uint()][src.base_offset.to_uint() +
        (ih * src.w.to_uint() + iw) * 12 + channel];
}

static void run_c12_stride1(esp_int8::window_sched_desc_t sched) {
    using namespace esp_int8;
    constexpr unsigned width = 16;
    constexpr unsigned height = 7;
    tensor_desc_t src = {};
    src.bank_id=0; src.elem_bytes=1; src.h=height; src.w=width; src.c=12;
    memory[0].assign(height * width * 12 + 64, 0);
    for (unsigned i=0; i<height*width*12; ++i) memory[0][i]=(i*29+17)&255;
    sched.out_w=width; sched.flags=1; sched.reserved[9]=0;
    csim_reset_narrow_transfer_words();
    for (unsigned oh=0; oh<height; ++oh) {
        hls::stream<act_vec_t> act0, act1;
        u8_t state=0;
        scheduled_window_generator_row(src,sched,act0,act1,u16_t(oh),state);
        for (unsigned issue=0; issue<width/2; ++issue) {
            for (int kt=0; kt<4; ++kt) {
                if (act0.empty() || act1.empty()) { ++failures; break; }
                const act_vec_t a=act0.read(), b=act1.read();
                for (int lane=0; lane<32; ++lane) {
                    if (a.range(lane*8+7,lane*8).to_uint()!=
                            expected_c12_lane(src,oh,issue*2,lane,kt) ||
                        b.range(lane*8+7,lane*8).to_uint()!=
                            expected_c12_lane(src,oh,issue*2+1,lane,kt)) ++failures;
                }
            }
        }
        if (!act0.empty() || !act1.empty() || state.to_uint()!=0U) ++failures;
    }
    const unsigned issues=width/2;
    const unsigned long long expected=
        static_cast<unsigned long long>(height)*(12U+(issues-1U)*6U);
    const unsigned long long transfer=csim_get_narrow_transfer_words();
    if (transfer!=expected) ++failures;
    std::printf("[C12-S1] transfer=%llu expected=%llu errors=%u\n",
                transfer,expected,failures);
}

int main() {
    const char* dir=std::getenv("ESP_INT8_CSIM_ARTIFACT_DIR");
    if (!dir) { std::printf("[FAIL] missing ESP_INT8_CSIM_ARTIFACT_DIR\n"); return 1; }
    std::ifstream in(std::string(dir)+"/PARAM.BIN",std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    if (bytes.size()<3904) return 1;
    const auto sched=load_schedule(bytes);
    if (sched.reserved[9].to_uint()!=(2U | (48U<<2))) return 1;
    if (run_rows(sched,"48 NONE",512,7,1,false,0,0,1)!=480) ++failures;
    if (run_rows(sched,"48 frame1",512,7,1,true,0,0,1)!=336) ++failures;
    if (run_rows(sched,"48 frame2 changed data",512,7,1,true,0,0,2)!=336) ++failures;
    if (run_rows(sched,"96 KEEP1",1024,7,1,true,0,0,3)!=672) ++failures;
    run_rows(sched,"branch A rows0..1",512,7,1,true,0,0,4,0,2);
    if (run_rows(sched,"branch B rows2..3",512,7,1,true,1,0,5,2,2)!=192) ++failures;
    run_rows(sched,"same bank A rows0..1",512,7,1,true,0,0,6,0,2);
    if (run_rows(sched,"base change rows2..3",512,7,1,true,0,4096,7,2,2)!=192) ++failures;
    if (run_rows(sched,"padding0 odd output tail",512,7,0,true,0,0,8)!=336) ++failures;
    run_c12_stride1(load_schedule_id(bytes,2));
    std::printf("[C3-REUSE] %s errors=%u\n",failures ? "FAIL" : "PASS",failures);
    return failures ? 1 : 0;
}
