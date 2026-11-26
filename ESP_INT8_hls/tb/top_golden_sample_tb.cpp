#include "top_call.hpp"
#include "../include/npu_ctrl.hpp"
#include "../../ESP_INT8_app/src/runtime_catalog.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace esp_int8 { unsigned csim_last_uop(); unsigned csim_last_error(); }
static bool read_binary(const std::string& path,std::vector<unsigned char>& bytes) {
    std::ifstream file(path,std::ios::binary);
    if (!file) { std::printf("[FAIL] missing file %s\n",path.c_str()); return false; }
    bytes.assign(std::istreambuf_iterator<char>(file),{});
    return !file.bad();
}
static void pack(const std::vector<unsigned char>& bytes,std::vector<esp_int8::axi_vec_t>& words) {
    std::fill(words.begin(),words.end(),esp_int8::axi_vec_t(0));
    for (unsigned b=0;b<bytes.size();++b) words[b/32].range((b%32)*8+7,(b%32)*8)=bytes[b];
}
int main() {
    const char* setting=std::getenv("ESP_INT8_CSIM_RUNTIME_ROOT");
    const std::string root=setting?setting:"D:/ESP_INT8/hw_artifacts/unified_int8";
    std::ifstream catalog(root+"/RUNS.CSV");
    std::string line;
    if (!std::getline(catalog,line)) return 1;
    if (!line.empty() && line.back()=='\r') line.pop_back();
    if (line!=RUNTIME_CATALOG_HEADER) return 2;
    std::vector<runtime_case_t> cases;
    while (std::getline(catalog,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        runtime_case_t c;
        if (cases.size()>=RUNTIME_MAX_CASES || !runtime_parse_case(line.c_str(),&c)) return 3;
        for (const auto& old:cases) if (!std::strcmp(old.key,c.key)) return 4;
        cases.push_back(c);
    }
    if (cases.empty()) return 5;
    std::vector<esp_int8::axi_vec_t> param(8192),frame(49152),out(16385);
    esp_int8::u8_t stage=0,pc=0,kind=0,win=0,sa=0,post=0;
    ap_uint<1> active=0;
    std::uint32_t abi=0,status=0,cap=0;
    auto call=[&](unsigned mode,unsigned pid,unsigned source,unsigned mask,unsigned count=75) {
        espnet_encoder_int8_core(frame.data(),out.data(),param.data(),mode,count,
            stage,active,pc,kind,win,sa,post,pid,abi,status,source,mask,cap);
    };
    call(esp_int8::MODE_RUN,0,0,0);
    if (!((status>>8)&255U) || stage!=esp_int8::PROF_STAGE_ERROR) return 6;
    call(esp_int8::MODE_INIT,8,0,0);
    if ((status&1U) || abi!=esp_int8::RUNTIME_ABI_ID || cap!=esp_int8::RUNTIME_CAPABILITY) return 7;
    for (unsigned index=0;index<=cases.size();++index) {
        const runtime_case_t& c=cases[index%cases.size()];
        runtime_profile_t p; runtime_io_t io;
        if (!runtime_profile_lookup(c.profile,&p) || !runtime_io_lookup(c.profile,c.source,c.mask,&io)) return 8;
        std::vector<unsigned char> parameters,input,expected_logits,expected_mask;
        const std::string dir=root+"/"+c.key+"/";
        if (!read_binary(root+"/"+c.param,parameters) || !read_binary(root+"/"+c.input,input) ||
            !read_binary(dir+"LOGITS.BIN",expected_logits) || !read_binary(dir+"MASK.BIN",expected_mask)) return 9;
        if (parameters.size()>262144 || parameters.size()<128 || input.size()!=io.input_bytes ||
            expected_logits.size()!=p.logits_h*p.logits_w*c.classes || expected_mask.size()!=io.mask_bytes ||
            runtime_crc32(parameters.data(),parameters.size())!=c.param_crc ||
            runtime_crc32(input.data(),input.size())!=c.input_crc ||
            runtime_crc32(expected_mask.data(),expected_mask.size())!=c.mask_crc) return 10;
        pack(parameters,param); pack(input,frame);
        std::fill(out.begin(),out.end(),esp_int8::axi_vec_t(0x123456));
        call(esp_int8::MODE_INIT,c.profile,c.source,c.mask);
        if (status!=((c.profile<<16)|1U) || stage!=esp_int8::PROF_STAGE_IDLE) return 11;
        call(esp_int8::MODE_RUN,c.profile,7,c.mask);
        if (!((status>>8)&255U) || stage!=esp_int8::PROF_STAGE_ERROR) return 12;
        call(0x11U,c.profile,c.source,c.mask); // Old upper-bit mode aliases must not execute.
        if (!((status>>8)&255U) || stage!=esp_int8::PROF_STAGE_ERROR) return 13;
        call(esp_int8::MODE_RUN,c.profile,c.source,c.mask,0x1004bU);
        if (!((status>>8)&255U) || stage!=esp_int8::PROF_STAGE_ERROR) return 14;
        std::remove("csim_u72_lowres_logits.bin");
        std::printf("[UNIFIED-CSIM] begin %s source=%u mask=%u\n",c.key,c.source,c.mask);
        std::fflush(stdout);
        call(esp_int8::MODE_RUN,c.profile,c.source,c.mask);
        if (abi!=esp_int8::RUNTIME_ABI_ID || cap!=esp_int8::RUNTIME_CAPABILITY ||
            status!=((c.profile<<16)|1U) || stage!=esp_int8::PROF_STAGE_IDLE || active) {
            std::printf("[FAIL] status=%x stage=%u uop=%u error=%u\n",status,stage.to_uint(),
                esp_int8::csim_last_uop(),esp_int8::csim_last_error()); return 15;
        }
        std::vector<unsigned char> actual_logits;
        if (!read_binary("csim_u72_lowres_logits.bin",actual_logits) ||
            actual_logits.size()!=expected_logits.size()) return 16;
        unsigned logits_errors=0,mask_errors=0;
        for (unsigned b=0;b<expected_logits.size();++b) logits_errors+=actual_logits[b]!=expected_logits[b];
        for (unsigned b=0;b<io.mask_bytes;++b)
            mask_errors+=out[b/32].range((b%32)*8+7,(b%32)*8).to_uint()!=expected_mask[b];
        for (unsigned word=(io.mask_bytes+31)/32;word<out.size();++word)
            if (out[word]!=0x123456) { std::printf("[FAIL] write beyond active output\n"); return 17; }
        std::printf("[UNIFIED-CSIM] %s logits=%u/%zu mask=%u/%zu\n",c.key,logits_errors,
            expected_logits.size(),mask_errors,expected_mask.size()); std::fflush(stdout);
        if (logits_errors || mask_errors) return 18;
    }
    std::printf("[UNIFIED-CSIM] PASS runs=%zu\n",cases.size()+1);
    return 0;
}
