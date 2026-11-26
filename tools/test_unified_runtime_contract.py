"""Unified INT8 runtime/layout gates; does not launch HLS or change artifacts."""

from dataclasses import replace
from pathlib import Path
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_contract as geometry
import export_int8_hw_blob as compiler
import hw_param_replay as replay


class RuntimePublishingTests(unittest.TestCase):
    def test_metadata_capacity_covers_every_published_param_without_abi_changes(self):
        self.assertEqual((compiler.MAX_EXEC_PLAN_COUNT, compiler.MAX_FIXED_EXEC_DESC_COUNT), (16, 10))
        root = Path(__file__).resolve().parents[1]/'hw_artifacts/unified_int8'
        files = sorted(root.glob('*/PARAM.BIN'))
        self.assertEqual(len(files), 10)
        for path in files:
            with self.subTest(case=path.parent.name):
                header = struct.unpack_from('<32I', path.read_bytes())
                self.assertEqual(header[1], 4)
                self.assertEqual((header[2], *header[4:8], header[9], header[28], header[31],
                                  (header[27]-header[26])//32), (19, 26, 7, 14, 3, 16, 16, 26, 10))

    def test_source_gate_rejects_clones_and_pragma_crutches(self):
        from tools import check_p6_hls_structure as check
        self.assertTrue(hasattr(check,'check_unified_sources'),'unified source delivery gate missing')
        source=check.load_sources()
        self.assertEqual(check.check_unified_sources(source),[])
        bad=dict(source); bad['conv_engine.cpp']+='\n#pragma HLS ALLOCATION function instances=x limit=1'
        self.assertTrue(check.check_unified_sources(bad))
        bad=dict(source); bad['conv_engine.cpp']+='\nvoid clone(){ systolic_array_core_row(); }'
        self.assertTrue(check.check_unified_sources(bad))
        bad=dict(source); bad['memory.cpp']=bad['memory.cpp'].replace(
            '[FMBUF_URAM_AXI_WORDS]', '[4][FMBUF_URAM_AXI_WORDS/4]')
        self.assertTrue(check.check_unified_sources(bad), 'failed four-bank fabric must not return')

    def test_conv_task_configuration_has_one_common_descriptor_path(self):
        from tools import check_p6_hls_structure as check
        root=Path(__file__).resolve().parents[1]
        source=(root/'ESP_INT8_hls/src/conv_engine.cpp').read_text(encoding='utf-8-sig')
        body=check.function_body(source,'build_conv_task_from_issue')
        self.assertIsNotNone(body)
        for name in ('param_dma_get_conv_exec_desc','param_dma_get_window_sched',
                     'param_dma_get_conv_qparam','conv_cfg_from_exec_desc'):
            self.assertEqual(check.count_call_sites(body,name),1,name+' must be shared before mode adapters')
        for retired in ('build_normal_conv_task','build_block5_branch_task','block5_prev_branch','result.conv_desc'):
            self.assertNotIn(retired,source)

    def test_affine_cache_has_packed_bram_owner_instead_of_lane_auto_rams(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'ESP_INT8_hls/src/param_dma.cpp').read_text(encoding='utf-8-sig')
        self.assertIn('BIND_STORAGE variable=s_affine_qparam_words type=ram_2p impl=bram',source)
        self.assertIn('ARRAY_PARTITION variable=s_affine_qparam_words complete dim=1',source)
        self.assertIn('RESET variable=s_affine_qparam_words off',source)
        self.assertNotIn('static affine_qparam_t s_affine_qparam[',source)

    @unittest.skipUnless(os.environ.get('ESP_INT8_RUNTIME_RELEASE_CHECK'), 'opt-in published full-network checkpoint')
    def test_default_tb_all_published_models_in_one_process(self):
        root=Path(__file__).resolve().parents[1]
        catalog=root/'hw_artifacts/unified_int8'
        self.assertEqual(len(compiler.validate_runtime_catalog(catalog)),10)
        with tempfile.TemporaryDirectory(prefix='esp_release_') as directory:
            exe=Path(directory)/'release.exe'
            command=['D:/mingw64/bin/g++.exe','-std=c++14','-O2','-fmax-errors=2','-Wno-unknown-pragmas',
                '-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE','-ID:/Xilinx/2025.1/Vitis/include',
                '-I'+str(root/'ESP_INT8_hls/include'),str(root/'ESP_INT8_hls/tb/top_golden_sample_tb.cpp'),
                *map(str,sorted((root/'ESP_INT8_hls/src').glob('*.cpp'))),'-o',str(exe)]
            build=subprocess.run(command,capture_output=True,text=True,timeout=300)
            self.assertEqual(build.returncode,0,build.stderr)
            env={**os.environ,'ESP_INT8_CSIM_RUNTIME_ROOT':str(catalog)}
            run=subprocess.run([str(exe)],cwd=directory,env=env,capture_output=True,text=True,timeout=14400)
            print(run.stdout,flush=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('[UNIFIED-CSIM] PASS runs=11',run.stdout)

    def test_generated_app_contract_and_catalog_match_hls(self):
        self.assertTrue(hasattr(compiler,'render_runtime_app_contract'),'App contract generator missing')
        root=Path(__file__).resolve().parents[1]
        self.assertEqual((root/'ESP_INT8_app/src/runtime_profiles.h').read_text(),
                         compiler.render_runtime_app_contract())
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'runtime.c'
            out.write_text('''#include "runtime_catalog.h"
#include <stdio.h>
int main(void) {
 runtime_case_t c; runtime_profile_t p; runtime_io_t io;
 const char *valid="G240B2,0,2,0,0,G240B2/PARAM.BIN,G240B2/INPUTQ.BIN,G240B2/OUT.BIN,12345678,ABCDEF01,00000001";
 if (!runtime_parse_case(valid,&c) || c.param_crc!=0x12345678 || c.input_bytes!=57600) return 1;
 if (runtime_parse_case("BAD,0,2,0,1,A/P.BIN,A/I.BIN,A/O.BIN,0,0,0",&c)) return 2;
 if (runtime_parse_case("BAD,0,2,0,0,../P.BIN,A/I.BIN,A/O.BIN,0,0,0",&c)) return 3;
 if (runtime_parse_case("BAD,0,2,0,0,A/P.BIN,A/I.BIN,A/O.BIN,0,0,0,extra",&c)) return 4;
 if (runtime_crc32((const unsigned char*)"123456789",9)!=0xcbf43926U) return 5;
 for(unsigned id=0;id<8;++id) { if(!runtime_profile_lookup(id,&p)) return 6;
  for(unsigned s=0;s<7;++s) for(unsigned m=0;m<2;++m)
   if(runtime_io_lookup(id,s,m,&io)) printf("%u,%u,%u,%u,%u,%u,%u\\n",id,s,m,io.input_bytes,io.mask_bytes,io.roi_top,io.roi_left);
 }
 return 0;
}''')
            exe=Path(directory)/'runtime.exe'
            build=subprocess.run(['D:/mingw64/bin/gcc.exe','-std=c11','-Wall','-Wextra','-Werror',
                '-I'+str(root/'ESP_INT8_app/src'),str(out),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            expected=[]
            for p in geometry.RUNTIME_PROFILES:
                for s in range(7):
                    for m in range(2):
                        try: r=geometry.runtime_selection(p.profile_id,2,src_mode=s,mask_mode=m)
                        except ValueError: continue
                        expected.append(f'{p.profile_id},{s},{m},{r.input_bytes},{r.mask_bytes},{r.roi[0]},{r.roi[1]}')
            self.assertEqual(run.stdout.splitlines(),expected)

    def test_manifest_and_publication_bind_identity_and_preserve_frozen_numbers(self):
        self.assertTrue(hasattr(compiler,'publish_runtime_artifact'),'unified publisher is missing')
        root=Path(__file__).resolve().parents[1]
        old=root/'hw_artifacts/binary2_int8_h240w240_gray_r6_v4'
        raw=(old/'PARAM.BIN').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            frozen=Path(directory)/'FROZEN'; frozen.mkdir()
            with self.assertRaises(ValueError):
                compiler.publish_runtime_artifact(frozen,frozen/'CHILD')
            out=Path(directory)/'G240B2'
            manifest=compiler.publish_runtime_artifact(old,out)
            selection=geometry.runtime_selection_from_manifest(manifest)
            self.assertEqual(selection,geometry.runtime_selection(0,2))
            self.assertEqual((out/'INPUTQ.BIN').read_bytes(),(old/'input_q.bin').read_bytes())
            self.assertEqual((out/'LOGITS.BIN').read_bytes(),(old/'golden_output_q.bin').read_bytes())
            self.assertEqual((out/'MASK.BIN').read_bytes(),(old/'golden_fullres_mask.bin').read_bytes())
            self.assertEqual(manifest['provenance']['source_param_sha256'],hashlib.sha256(raw).hexdigest())
            compiler.validate_runtime_artifact(out)
            altered=bytearray((out/'INPUTQ.BIN').read_bytes()); altered[0]^=1
            (out/'INPUTQ.BIN').write_bytes(altered)
            with self.assertRaisesRegex(ValueError,'hash|identity'):
                compiler.validate_runtime_artifact(out)
            with self.assertRaises(ValueError):
                compiler.publish_runtime_artifact(old,out)
        self.assertEqual((old/'PARAM.BIN').read_bytes(),raw)

    def test_runtime_view_rejects_unbound_geometry_and_selects_same_rgb_pixel(self):
        self.assertTrue(hasattr(geometry,'select_runtime_input'),'runtime view helper is missing')
        selected=geometry.runtime_selection(1,20,src_mode=5,mask_mode=1)
        data=np.arange(selected.input_bytes,dtype=np.int64).astype(np.int8).reshape(selected.source_shape)
        output=geometry.select_runtime_input(data,selected)
        axes=np.asarray(geometry.nearest_half_pixel_indices(512,240))
        np.testing.assert_array_equal(output,data[axes[:,None],256+axes[None,:],:])
        manifest={'runtime':selected.to_manifest()}
        self.assertEqual(geometry.runtime_selection_from_manifest(manifest),selected)
        manifest['runtime']['roi_yxhw'][1]=0
        with self.assertRaises(ValueError): geometry.runtime_selection_from_manifest(manifest)
        with self.assertRaises(ValueError): geometry.select_runtime_input(data.view(np.uint8),selected)
        gray=geometry.runtime_selection(0,2)
        with self.assertRaises(ValueError):
            geometry.runtime_mask_from_logits(np.zeros((2,30,30),dtype=np.int8),gray)
        with self.assertRaises(ValueError):
            geometry.runtime_mask_from_logits(np.zeros((1,2,30,30),dtype=np.int8),gray)


@unittest.skipUnless(os.environ.get('ESP_INT8_FULL_RUNTIME_CASES'), 'opt-in full-network checkpoint')
class RuntimeNetworkTests(unittest.TestCase):
    @staticmethod
    def structural_param_fixture(source_path, selection, destination):
        """Test-only shape specialization, never a trained/released model artifact."""
        old = replay.ParamBlob(source_path)
        g = selection.profile.geometry
        if old.geometry.input_channels != g.input_channels or old.class_count != selection.classes:
            raise ValueError('structural fixture must preserve weight/channel/quantization identities')
        raw=bytearray(source_path.read_bytes())
        header=struct.unpack_from('<32I',raw)
        old_g=old.geometry
        def height(value):
            self_check=value*g.input_height
            if self_check%old_g.input_height: raise ValueError('nonintegral structural fixture height')
            return self_check//old_g.input_height
        def width(value):
            self_check=value*g.input_width
            if self_check%old_g.input_width: raise ValueError('nonintegral structural fixture width')
            return self_check//old_g.input_width
        tensors=compiler.build_tensor_plan(g,selection.classes,layout=compiler.build_runtime_memory_layout(g))
        raw[header[10]:header[10]+19*16]=b''.join(compiler.pack_tensor_desc(t) for t in tensors)
        for i,desc in enumerate(old.conv_exec_by_index):
            struct.pack_into('<2H',raw,header[22]+i*36+4,height(desc.in_h),width(desc.in_w))
        for i,desc in enumerate(old.window_sched_by_index):
            users=[conv for conv in old.conv_exec_by_index if conv.window_sched_id==i]
            if not users: raise ValueError('unused window record in structural fixture')
            compiled,_=compiler.make_window_pack_schedule(desc.in_c,desc.kernel,
                source_w=width(users[0].in_w),out_w=width(desc.out_w),stride=desc.stride,
                dilation=desc.dilation,padding=desc.padding,flags=desc.flags&~compiler.WINDOW_SCHED_FLAG_ODD_TAIL)
            compiled=replace(compiled,cmd_base=desc.cmd_base,cmd_count=desc.cmd_count,kt_cmd_base=desc.kt_cmd_base)
            raw[header[23]+i*128:header[23]+(i+1)*128]=compiler.pack_window_sched_desc(compiled)
        for i in range((header[27]-header[26])//32):
            base=header[26]+i*32
            h,w=struct.unpack_from('<2H',raw,base+8)
            struct.pack_into('<2H',raw,base+8,height(h),width(w))
        for i,desc in enumerate(old.block5_sched[:4]):
            struct.pack_into('<3H',raw,header[30]+i*32+16,height(desc.out_h),width(desc.out_w),32)
        uops=compiler.build_uops(selection.classes,geometry=g)
        data=b''.join(compiler.pack_uop(u) for u in uops)
        raw[header[16]:header[16]+len(data)]=data
        original=source_path.read_bytes()
        if raw[header[17]:header[16]]!=original[header[17]:header[16]]:
            raise AssertionError('structural fixture changed weights or qparams')
        destination.write_bytes(raw)

    @staticmethod
    def array_replay_reference(blob, input_path):
        """Existing replay with exact float64 convolution of bounded integer products."""
        torch=compiler.torch
        def exact_conv(model, param_id, array):
            desc=model.conv_exec[param_id]
            k=desc.kernel*desc.kernel*desc.in_c
            weight=np.stack([np.concatenate([model.get_packed_weight_tile(param_id,oc,kt)
                for kt in range(desc.k_tiles)])[:k] for oc in range(desc.out_c)])
            weight=weight.reshape(desc.out_c,desc.kernel,desc.kernel,desc.in_c).transpose(0,3,1,2).copy()
            tensor=torch.from_numpy(array.transpose(2,0,1).copy()).to(torch.float64)[None]
            with torch.no_grad():
                accum=torch.nn.functional.conv2d(tensor,torch.from_numpy(weight).to(torch.float64),
                    stride=desc.stride,padding=desc.padding,dilation=desc.dilation)
            accum=accum[0].permute(1,2,0).numpy().astype(np.int32)
            bias,mult,shift=model.conv_qparam[desc.qparam_id]
            result=np.empty(accum.shape,dtype=np.int8)
            for c in range(desc.out_c):
                result[:,:,c]=replay.requant_i32_to_i8(accum[:,:,c],bias[c],mult[c],shift[c],
                    replay.ACT_RELU if desc.flags&replay.FLAG_RELU_EN else replay.ACT_NONE)
            return result
        # All sums are <2^31 integers, exactly represented by float64; no floating requantization.
        with mock.patch.object(replay,'replay_conv_array',exact_conv):
            return replay.replay_prefix(blob,input_path,74).tensors[17]

    def test_frozen_models_and_lite_paths_in_one_top_process(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "TOP_CALL_HEADER"
#include "npu_ctrl.hpp"
#include "npu_uop.hpp"
#include <fstream>
#include <iterator>
#include <vector>
#include <cstdio>
namespace esp_int8 { unsigned csim_last_error(); unsigned csim_last_uop(); }
static std::vector<unsigned char> load(const char* path) {
  std::ifstream file(path,std::ios::binary);
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)),{});
}
static void pack(const std::vector<unsigned char>& bytes,std::vector<esp_int8::axi_vec_t>& out) {
  for(unsigned i=0;i<bytes.size();++i) out[i/32].range((i%32)*8+7,(i%32)*8)=bytes[i];
}
int main(int argc,char** argv) {
  using namespace esp_int8;
  for(int arg=1;arg+8<argc;arg+=9) {
    unsigned profile=std::strtoul(argv[arg],nullptr,10), classes=std::strtoul(argv[arg+1],nullptr,10);
    unsigned source=std::strtoul(argv[arg+2],nullptr,10), mask=std::strtoul(argv[arg+3],nullptr,10);
    runtime_io_t io; runtime_profile_t p;
    if(!runtime_io_lookup(profile,source,mask,io) || !runtime_profile_lookup(profile,p)) return 1;
    auto parameter=load(argv[arg+4]), input=load(argv[arg+5]);
    auto expected_logits=load(argv[arg+6]), expected_mask=load(argv[arg+7]);
    unsigned tag=std::strtoul(argv[arg+8],nullptr,10);
    if(input.size()!=io.input_bytes || expected_logits.size()!=p.logits_h*p.logits_w*classes
       || expected_mask.size()!=io.mask_bytes || parameter.size()>262144) return 2;
    std::vector<axi_vec_t> param(8192), frame(input.size()/32), output(io.mask_bytes/32+1);
    pack(parameter,param); pack(input,frame); output.back()=0x123456;
    u8_t stage=0,pc=0,kind=0,win=0,sa=0,post=0; ap_uint<1> active=0;
    std::uint32_t abi=0,status=0,cap=0;
    auto call=[&](unsigned mode) {
      espnet_encoder_int8_core(frame.data(),output.data(),param.data(),mode,75,
          stage,active,pc,kind,win,sa,post,profile,abi,status,source,mask,cap);
    };
    std::printf("[FULL-CSIM] begin case=%u profile=%u classes=%u source=%u mask=%u\n",tag,profile,classes,source,mask);
    std::fflush(stdout); call(MODE_INIT);
    if((status&0xffff)!=1 || abi!=RUNTIME_ABI_ID || cap!=RUNTIME_CAPABILITY) return 3;
    call(MODE_RUN);
    if((status&0xffff)!=1 || stage!=PROF_STAGE_IDLE || csim_last_error()!=ERR_NONE) {
      std::printf("[FAIL] status=%x stage=%u logical_uop=%u error=%u\n",status,stage.to_uint(),csim_last_uop(),csim_last_error());
      return 4;
    }
    auto actual_logits=load("csim_u72_lowres_logits.bin");
    unsigned logit_errors=0,mask_errors=0;
    if(actual_logits.size()!=expected_logits.size()) return 5;
    for(unsigned b=0;b<expected_logits.size();++b) if(actual_logits[b]!=expected_logits[b]) {
      if(logit_errors<8) std::printf("logit mismatch b=%u got=%d expected=%d\n",b,int((signed char)actual_logits[b]),int((signed char)expected_logits[b]));
      ++logit_errors;
    }
    for(unsigned b=0;b<io.mask_bytes;++b)
      if(output[b/32].range((b%32)*8+7,(b%32)*8).to_uint()!=expected_mask[b]) ++mask_errors;
    if(output.back()!=0x123456) return 6;
    std::printf("[FULL-CSIM] case=%u logits=%u/%zu mask=%u/%zu\n",tag,logit_errors,expected_logits.size(),mask_errors,expected_mask.size());
    std::fflush(stdout);
    if(logit_errors || mask_errors) return 7;
  }
  return 0;
}
'''.replace('TOP_CALL_HEADER', (root/'ESP_INT8_hls/tb/top_call.hpp').as_posix())
        cases = ((0,2,'binary2_int8_h240w240_gray_r6_v4',0,0),
                 (3,2,'binary2_int8_h256w512_r2_v4',0,0),
                 (3,20,'cityscapes20_int8_h256w512_r2_v4',0,0),
                 (4,2,'binary2_int8_h512w512_gray_finetuned_v4',0,0),
                 (0,2,'binary2_int8_h240w240_gray_r6_v4',2,1),
                 (7,20,'cityscapes20_int8_h256w512_r2_v4',0,0),
                 (1,20,'cityscapes20_int8_h256w512_r2_v4',6,1))
        with tempfile.TemporaryDirectory(prefix='esp_unified_full_') as directory:
            folder = Path(directory)
            path, binary = folder/'network.cpp', folder/'network.exe'
            path.write_text(source, encoding='ascii')
            arguments = []
            for index in map(int,os.environ['ESP_INT8_FULL_RUNTIME_CASES'].split(',')):
                pid,classes,name,src,mask = cases[index]
                selection = geometry.runtime_selection(pid,classes,src_mode=src,mask_mode=mask)
                g = selection.profile.geometry
                artifact = root/'hw_artifacts'/name
                param = folder/f'p{index}.bin'
                input_path = artifact/'input_q.bin'
                source_blob=replay.ParamBlob(artifact/'PARAM.BIN')
                fixture=source_blob.geometry!=g
                if fixture:
                    self.structural_param_fixture(artifact/'PARAM.BIN',selection,param)
                    source_array=np.fromfile(input_path,dtype=np.int8).reshape(source_blob.geometry.input_shape_nhwc)
                    native=(source_array.repeat(2,axis=0).repeat(2,axis=1) if pid==7
                            else source_array[8:248,136:376].copy())
                    input_path=folder/f'native{index}.bin'
                    native.tofile(input_path)
                    # First prove the faster independent reference against the original frozen model.
                    reference=self.array_replay_reference(source_blob,artifact/'input_q.bin')
                    np.testing.assert_array_equal(reference,np.fromfile(artifact/'golden_output_q.bin',dtype=np.int8).reshape(reference.shape))
                    logits_path=folder/f'golden{index}.bin'
                    self.array_replay_reference(replay.ParamBlob(param),input_path).tofile(logits_path)
                else:
                    param.write_bytes(compiler.recompile_runtime_param((artifact/'PARAM.BIN').read_bytes(),selection))
                    logits_path = artifact/'golden_output_q.bin'
                if src:
                    native = np.fromfile(input_path,dtype=np.uint8).reshape(240,240,g.input_channels)
                    array = np.zeros(selection.source_shape,dtype=np.uint8)
                    ys=np.asarray(geometry.nearest_half_pixel_indices(selection.roi[2],240))+selection.roi[0]
                    xs=np.asarray(geometry.nearest_half_pixel_indices(selection.roi[3],240))+selection.roi[1]
                    array[ys[:,None],xs[None,:],:]=native
                    input_path=folder/f'i{index}.bin'
                    array.tofile(input_path)
                logits = np.fromfile(logits_path,dtype=np.int8).reshape(g.logits_height,g.logits_width,classes).astype(np.int64)
                mask_array=np.empty(selection.mask_shape,dtype=np.uint8)
                for y,(y0,y1,fy,dy) in enumerate(geometry.linear_half_pixel_axis(g.logits_height,selection.mask_shape[0])):
                    for x,(x0,x1,fx,dx) in enumerate(geometry.linear_half_pixel_axis(g.logits_width,selection.mask_shape[1])):
                        scores=((logits[y0,x0]*(dx-fx)+logits[y0,x1]*fx)*(dy-fy)
                                +(logits[y1,x0]*(dx-fx)+logits[y1,x1]*fx)*fy)
                        mask_array[y,x]=scores.argmax()
                mask_path=folder/f'm{index}.bin'
                mask_array.tofile(mask_path)
                arguments.extend(map(str,(pid,classes,src,mask,param,input_path,logits_path,mask_path,index)))
            command = [shutil.which('g++'), '-std=c++14', '-O2', '-Wno-unknown-pragmas',
                       '-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE', '-ID:/Xilinx/2025.1/Vitis/include',
                       '-I'+str(root/'ESP_INT8_hls/include'), str(path),
                       *[str(p) for p in sorted((root/'ESP_INT8_hls/src').glob('*.cpp'))], '-o', str(binary)]
            built = subprocess.run(command,capture_output=True,text=True,timeout=300)
            self.assertEqual(built.returncode,0,built.stderr)
            run = subprocess.run([str(binary),*arguments],cwd=folder,capture_output=True,text=True,timeout=14400)
            print(run.stdout,flush=True)
            self.assertEqual(run.returncode,0,f'Full-network case failed: {run.returncode}\n{run.stdout}\n{run.stderr}')


class RuntimeSelectionTests(unittest.TestCase):
    def test_input_fifo_token_contract_is_bounded_and_acyclic(self):
        for profile in geometry.RUNTIME_PROFILES:
            for mode in range(7):
                try:
                    selected = geometry.runtime_selection(profile.profile_id,2,src_mode=mode)
                except ValueError:
                    continue
                output_count=profile.geometry.input_bytes//32
                if mode==0:
                    input_count=output_count
                    actions=['r','w']*output_count
                else:
                    channels=selected.source_shape[2]
                    side=selected.roi[2]
                    input_count=240*side*channels//32
                    actions=[]
                    upper=False
                    carry=phase=0
                    for _ in range(240*16):
                        for _ in range(channels*(2 if side==512 else 1)):
                            if not upper:
                                actions.append('r')
                            upper=not upper
                        pending=15*channels+carry
                        for _ in range(pending//4):
                            phase+=1
                            if phase==8:
                                actions.append('w')
                                phase=0
                        carry=pending%4
                    self.assertEqual((upper,carry,phase),(False,0,0))
                self.assertEqual((actions.count('r'),actions.count('w')),(input_count,output_count))
                # Blocking queues with the production capacities and bounded sink stalls.
                produced=consumed=cursor=q0=q1=0
                ticks=stuck=0
                while consumed<output_count:
                    progress=False
                    if q1 and ticks%7 not in (0,1,2):
                        q1-=1
                        consumed+=1
                        progress=True
                    if cursor<len(actions):
                        op=actions[cursor]
                        if (op=='r' and q0) or (op=='w' and q1<4):
                            if op=='r': q0-=1
                            else: q1+=1
                            cursor+=1
                            progress=True
                    if produced<input_count and q0<16:
                        q0+=1
                        produced+=1
                        progress=True
                    ticks+=1
                    stuck=0 if progress else stuck+1
                    self.assertLess(stuck,8,'finite-token input DAG deadlocked')
                self.assertEqual((produced,cursor,q0,q1),(input_count,len(actions),0,0))
                # Prove the fixed row-emission bound for every native/ROI scale.
                rows=geometry.linear_half_pixel_axis(profile.geometry.logits_height,selected.mask_shape[0])
                owner_counts=np.bincount([hi for _,hi,_,_ in rows],minlength=profile.geometry.logits_height)
                self.assertLessEqual(int(owner_counts.max()),32)

    def profiles(self):
        self.assertTrue(hasattr(geometry, "RUNTIME_PROFILES"), "runtime profile registry is missing")
        return geometry.RUNTIME_PROFILES

    def test_exactly_eight_stable_profile_ids(self):
        profiles = self.profiles()
        self.assertEqual([p.profile_id for p in profiles], list(range(8)))
        self.assertEqual({p.geometry.input_shape_nhwc for p in profiles}, {
            (h, w, c) for h, w in ((240, 240), (256, 512), (512, 512), (512, 1024))
            for c in (1, 3)
        })

    def test_native_profiles_have_exact_source_and_mask(self):
        for profile in self.profiles():
            for classes in (2, 20):
                run = geometry.runtime_selection(profile.profile_id, classes)
                self.assertEqual(run.source_shape, profile.geometry.input_shape_nhwc)
                self.assertEqual(run.mask_shape, run.source_shape[:2])
                self.assertEqual(run.input_bytes, profile.geometry.input_bytes)
                self.assertEqual(run.mask_bytes, profile.geometry.output_mask_bytes)

    def test_lite_source_ids_preserve_gray_abi_and_reject_wrong_encoder(self):
        profiles = self.profiles()
        gray, rgb = profiles[:2]
        for mode, shape in ((1, (256, 512, 1)), (2, (512, 1024, 1)), (3, (512, 512, 1)),
                            (4, (256, 512, 3)), (5, (512, 1024, 3)), (6, (512, 512, 3))):
            profile = gray if shape[2] == 1 else rgb
            run = geometry.runtime_selection(profile.profile_id, 2, src_mode=mode, mask_mode=1)
            self.assertEqual(run.source_shape, shape)
            side = min(shape[:2])
            self.assertEqual(run.roi, (0, (shape[1] - side)//2, side, side))
            self.assertEqual(run.mask_shape, (side, side))
            low = geometry.runtime_selection(profile.profile_id, 2, src_mode=mode)
            self.assertEqual(low.mask_shape, (240, 240))
            with self.assertRaises(ValueError):
                geometry.runtime_selection((rgb if shape[2] == 1 else gray).profile_id, 2, src_mode=mode)
            with self.assertRaises(ValueError):
                geometry.runtime_selection(profiles[2].profile_id, 2, src_mode=mode)

    def test_invalid_values_do_not_fall_back_to_native(self):
        self.profiles()
        for kwargs in (dict(profile_id=8), dict(classes=3), dict(src_mode=7),
                       dict(mask_mode=2), dict(profile_id=4, mask_mode=1),
                       dict(profile_id=-1), dict(profile_id=0.5)):
            args = dict(profile_id=0, classes=2)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                geometry.runtime_selection(**args)

    def test_integer_half_pixel_sampling_and_interpolation(self):
        self.profiles()
        for source, output in ((256, 240), (512, 240)):
            expected = [((2*d + 1)*source)//(2*output) for d in range(output)]
            self.assertEqual(list(geometry.nearest_half_pixel_indices(source, output)), expected)
        for logits, output, denominator in ((30, 240, 16), (30, 256, 256), (30, 512, 512),
                                            (64, 512, 16), (4, 6, 6), (1, 240, 480)):
            axis = geometry.linear_half_pixel_axis(logits, output)
            self.assertEqual(len(axis), output)
            self.assertEqual(axis[0], (0, 0, 0, denominator))
            self.assertEqual(axis[-1], (logits-1, logits-1, 0, denominator))
            for d, (lo, hi, frac, den) in enumerate(axis):
                coordinate = min(max((d+0.5)*logits/output-0.5, 0), logits-1)
                self.assertAlmostEqual(lo + frac/den, coordinate)
                self.assertTrue(0 <= lo <= hi < logits)


class RuntimeMemoryTests(unittest.TestCase):
    def layout(self, h=512, w=1024, c=3):
        self.assertTrue(hasattr(compiler, "build_runtime_memory_layout"), "shared runtime allocator is missing")
        return compiler.build_runtime_memory_layout(geometry.DeploymentGeometry(h, w, 8, c))

    def test_maximum_layout_uses_shared_split_and_phase_scratch(self):
        layout = self.layout()
        self.assertEqual(layout.uram_bytes, 0x380000)
        self.assertEqual(layout.fmbuf_bytes, 0x530000)
        self.assertEqual(layout.pool1_base, 0x418000)
        self.assertEqual(layout.row_group_rows, 32)
        self.assertEqual(layout.b2_backup_rows, 64)
        self.assertEqual(layout.b2_backup_base, layout.l2b0_c1_base)
        for profile in geometry.RUNTIME_PROFILES:
            current = compiler.build_runtime_memory_layout(profile.geometry)
            self.assertEqual(current.uram_bytes, layout.uram_bytes)
            self.assertEqual(current.fmbuf_bytes, layout.fmbuf_bytes)
            self.assertGreaterEqual(current.pool1_base, layout.uram_bytes)

    def test_all_profile_lifetimes_and_reverse_row_handoffs(self):
        self.layout()
        for profile in geometry.RUNTIME_PROFILES:
            for classes in (2, 20):
                layout = compiler.build_runtime_memory_layout(profile.geometry)
                audit = compiler.audit_runtime_memory(profile.geometry, classes, layout=layout)
                self.assertEqual(audit["unsafe_overlaps"], [])
                self.assertTrue(audit["row_handoffs_safe"])
                self.assertTrue(audit["regions_in_bounds"])
                self.assertLessEqual(audit["backup_peak_rows"], layout.b2_backup_rows)

    def test_audit_rejects_missing_backup_and_live_pool_alias(self):
        layout = self.layout()
        g = geometry.DeploymentGeometry(512, 1024, 8, 3)
        for bad in (replace(layout, b2_backup_rows=0), replace(layout, pool1_base=0x3E0000),
                    replace(layout, l2b0_c1_base=layout.residual_base),
                    replace(layout, l3b0_block5_base=layout.l30_base),
                    replace(layout, l2_scratch_slot_bytes=0), replace(layout, pool2_bytes=0)):
            with self.assertRaises(ValueError):
                compiler.audit_runtime_memory(g, 20, layout=bad)

    def test_tensor_addresses_and_replay_pool_mapping_are_layout_owned(self):
        layout = self.layout()
        g = geometry.DeploymentGeometry(512, 1024, 8, 3)
        tensors = compiler.build_tensor_plan(g, 20, layout=layout)
        self.assertEqual(tensors[2].base_offset, layout.b1_base)
        self.assertEqual(tensors[11].base_offset, layout.l30_base)
        memory = replay.FeatureMemory(layout=layout)
        desc = replay.TensorDesc(bank_id=0x81, h=1, w=1, c=3, reserved0=3)
        values = np.array([-128, 17, 127], dtype=np.int8)
        memory.store_tile(desc, 0, 0, 0, 3, values)
        np.testing.assert_array_equal(memory.buf[layout.pool1_base:layout.pool1_base+3].view(np.int8), values)
        self.assertEqual(memory.buf.size, layout.fmbuf_bytes)

    def test_reverse_concat_preserves_actual_source_bytes_in_one_physical_ram(self):
        self.layout()
        for profile in geometry.RUNTIME_PROFILES:
            layout = compiler.build_runtime_memory_layout(profile.geometry)
            tensors = compiler.build_tensor_plan(profile.geometry, 2, layout=layout)
            memory = replay.FeatureMemory(layout=layout)
            sources, dst = (tensors[7], tensors[5], tensors[8]), tensors[9]
            rng = np.random.default_rng(profile.profile_id)
            originals = [rng.integers(0, 256, (s.h, s.w, s.c), dtype=np.uint8) for s in sources]
            for source, data in zip(sources, originals):
                for row in range(source.h):
                    start = memory.physical_base(source)+row*source.row_stride_bytes
                    memory.buf[start:start+source.w*source.c] = data[row].reshape(-1)
            saved = set()
            for row in reversed(range(dst.h)):
                current = []
                for index, source in enumerate(sources):
                    start = memory.physical_base(source)+row*source.row_stride_bytes
                    if index == 1 and row in saved:
                        start = layout.b2_backup_base+(row & (layout.b2_backup_rows-1))*source.row_stride_bytes
                    current.append(memory.buf[start:start+source.w*source.c].copy().reshape(source.w, source.c))
                saved.discard(row)
                start = dst.base_offset+row*dst.row_stride_bytes
                end = start+dst.row_stride_bytes
                residual = sources[1]
                for future in range(row):
                    lo = residual.base_offset+future*residual.row_stride_bytes
                    hi = lo+residual.row_stride_bytes
                    if start < hi and lo < end and future not in saved:
                        target = layout.b2_backup_base+(future & (layout.b2_backup_rows-1))*residual.row_stride_bytes
                        memory.buf[target:target+residual.row_stride_bytes] = memory.buf[lo:hi].copy()
                        saved.add(future)
                packed = np.concatenate(current, axis=1).reshape(-1)
                memory.buf[start:end] = 0
                memory.buf[start:start+packed.size] = packed
            expected = np.concatenate(originals, axis=2)
            for row in range(dst.h):
                start = dst.base_offset+row*dst.row_stride_bytes
                actual = memory.buf[start:start+dst.w*dst.c].reshape(dst.w, dst.c)
                np.testing.assert_array_equal(actual, expected[row])
            self.assertEqual(saved, set())

    def test_budget_reports_headroom_without_claiming_synthesis(self):
        self.layout()
        budget = compiler.runtime_memory_budget()
        self.assertEqual(budget["uram"], 112)
        self.assertEqual(budget["fmbuf_bram18k"], 864)
        self.assertGreater(budget["other_bram18k_reserve"], 0)
        self.assertLessEqual(budget["bram18k_upper_bound"], 1488)
        self.assertEqual(budget["evidence"], "pre-synthesis estimate; requires csynth/implementation")

    def test_read_only_audit_cli_never_creates_export_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'must_not_exist'
            result = subprocess.run([sys.executable, '-B', str(Path(compiler.__file__)),
                                     '--audit-runtime-memory', '--out-dir', str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('pre_csynth_estimate=1 physical_resource_gate_pending=1', result.stdout)
            self.assertIn('checked_profile_class_pairs=16', result.stdout)
            self.assertFalse(output.exists())

    def test_existing_native512_layout_does_not_change(self):
        g = geometry.DeploymentGeometry(512, 512, 8, 1)
        layout = compiler.build_memory_layout(g)
        self.assertEqual((layout.fmbuf_bytes, layout.uram_bytes), (0x378000, 0x358000))

    def test_frozen_inputs_and_integer_goldens_are_not_replaced(self):
        hashes = {
            'binary2_int8_h240w240_gray_r6_v4': (
                'c1f3b03245d4bc6514c2a2dfcada898453f0d8eca6292ddcd46b40c1c2b5dd87',
                '5889a10dc0bdfc8ab33e1464cc0f7e87a7a85a48b3c8ca05eac918a1e0acbd24'),
            'binary2_int8_h256w512_r2_v4': (
                '5a1f93c1247c7de7c6d499f864f1c8af3f8c2931774cc0a1d338b10246295215',
                '69cd033df44f1aa03cc70fba2f0dcdc67390afe4c6fd6a2ff833887a9dd800e4'),
            'cityscapes20_int8_h256w512_r2_v4': (
                '5a1f93c1247c7de7c6d499f864f1c8af3f8c2931774cc0a1d338b10246295215',
                'a36349f75fbb99ab515e4b29a92ecca7eb94559c3a26603a719017f9b0002332'),
            'binary2_int8_h512w512_gray_finetuned_v4': (
                'b3355dbbd88d11bf34bc984d5591b83a04951a1cf93b7a674c2e908d37bcfe93',
                '035a9f3cccd0f51379eed5c5749b3f33efdf9fd235bb04bc0178dc8f4f92460c'),
        }
        for folder, expected in hashes.items():
            root = Path(__file__).resolve().parents[1]/'hw_artifacts'/folder
            for filename, digest in zip(('input_q.bin', 'golden_output_q.bin'), expected):
                self.assertEqual(hashlib.sha256((root/filename).read_bytes()).hexdigest(), digest)

    def test_runtime_recompile_preserves_all_model_arithmetic_bytes(self):
        self.assertTrue(hasattr(compiler, "recompile_runtime_param"), "runtime PARAM recompiler is missing")
        cases = (
            (0, 2, 'binary2_int8_h240w240_gray_r6_v4', '9e380cebe305c4945642c4a334f75a7a2ba94eefa1f2ccc7f2964f8d3bf9d878'),
            (3, 2, 'binary2_int8_h256w512_r2_v4', 'd2de465a97c802d69b691bd852e3250e13eae1329ce67df14e20fceaeeb507e4'),
            (3, 20, 'cityscapes20_int8_h256w512_r2_v4', 'fe8f9541edcffcb8100582cc7a3b3b0351395a5c18d504f697a24acaed6e71df'),
            (4, 2, 'binary2_int8_h512w512_gray_finetuned_v4', '7c2175418f1f34e53bb3f12f57f8e182df204d27b2df0a3e136ac488baf45d4a'),
        )
        for pid, classes, folder, expected_hash in cases:
            path = Path(__file__).resolve().parents[1]/'hw_artifacts'/folder/'PARAM.BIN'
            original = path.read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), expected_hash)
            selection = geometry.runtime_selection(pid, classes)
            converted = compiler.recompile_runtime_param(original, selection)
            before, after = bytearray(original), bytearray(converted)
            header = struct.unpack_from('<32I', original)
            table, block = header[10], header[30]
            before[table:table+19*16] = after[table:table+19*16] = b'\0'*(19*16)
            for i in range(4):
                row_group = block+i*32+20
                self.assertEqual(struct.unpack_from('<H', converted, row_group)[0], 32)
                before[row_group:row_group+2] = after[row_group:row_group+2] = b'\0\0'
            for i in range(header[28]):
                transport = header[23]+i*128+126
                old_word = struct.unpack_from('<H', original, transport)[0]
                new_word = struct.unpack_from('<H', converted, transport)[0]
                self.assertEqual((old_word ^ new_word) & ~0x600, 0)
                before[transport:transport+2] = after[transport:transport+2] = b'\0\0'
            self.assertEqual(before, after, 'weights, quantization and numerical schedules must not change')
            with tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory)/'candidate.bin'
                candidate.write_bytes(converted)
                parsed = replay.ParamBlob(candidate)
                self.assertEqual(parsed.class_count, classes)
                self.assertEqual(parsed.geometry.input_shape_nhwc, selection.profile.geometry.input_shape_nhwc)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                compiler.recompile_runtime_param(original, geometry.runtime_selection(7, classes))
            with self.assertRaises(ValueError):
                compiler.recompile_runtime_param(original[:128], selection)


class RuntimeHlsProfileTests(unittest.TestCase):
    def test_ppu_common_compact_tail_preserves_c19_and_existing_layouts(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "PPU_SOURCE"
#include <iostream>
int main() {
  using namespace esp_int8; runtime_select_profile(1);
  const unsigned counts[]={12,16,17,19,25,28};
  const unsigned layouts[]={STORE_LAYOUT_COMPACT_C12,STORE_LAYOUT_COMPACT_C16,
    STORE_LAYOUT_COMPACT_C17,STORE_LAYOUT_COMPACT_C19,STORE_LAYOUT_COMPACT_C25,STORE_LAYOUT_COMPACT_C28};
  for(unsigned shape=0;shape<6;++shape) for(unsigned width=1;width<=256;++width) {
    if(width>65 && width<255) continue;
    unsigned channels=counts[shape];
    tensor_desc_t dst; dst.bank_id=0; dst.elem_bytes=1; dst.base_offset=32;
    dst.h=2; dst.w=width; dst.c=channels; dst.reserved0=channels|TENSOR_ROW_ALIGN32; dst.reserved1=0;
    row_consumer_desc_t consumer; consumer.mode=ROW_CONSUMER_NONE; consumer.valid_c=channels;
    consumer.reserved0=layouts[shape]; consumer.reserved1=0;
    conv_cfg_t cfg; cfg.in_h=2; cfg.in_w=width; cfg.out_c=channels; cfg.stride=1;
    hls::stream<act_vec_t> input;
    for(unsigned x=0;x<width;++x) {
      act_vec_t word=0;
      for(unsigned c=0;c<32;++c) word.range(c*8+7,c*8)=(x*37+c*11+19)&255;
      input.write(word);
    }
    tensor_desc_t unused; add_q_t unused_q; u8_t status=0;
    ppu_consume_conv_stream(consumer,dst,unused,false,unused_q,cfg,1,input,status);
    if(status!=1 || !input.empty()) return 1;
    unsigned stride=tensor_row_stride_bytes(dst).to_uint();
    for(unsigned wi=0;wi<stride/32;++wi) {
      act_vec_t word;
      if(!on_chip_memory_read_fmbuf_abs_word(0,32+stride+wi*32,word)) return 2;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned b=wi*32+lane;
        unsigned expected=b<width*channels?((b/channels)*37+(b%channels)*11+19)&255:0;
        if(word.range(lane*8+7,lane*8).to_uint()!=expected) {
          std::cerr<<"compact "<<channels<<" "<<width<<" "<<b; return 3;
        }
      }
    }
  }
  return 0;
}
'''.replace('PPU_SOURCE', (root/'ESP_INT8_hls/src/ppu.cpp').as_posix())
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'compact.cpp', Path(directory)/'compact.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), *[str(root/'ESP_INT8_hls/src'/name) for name in
                       ('memory.cpp','conv_store.cpp','param_dma.cpp','upsample_unit.cpp')], '-o', str(binary)]
            built = subprocess.run(command,capture_output=True,text=True,timeout=120)
            self.assertEqual(built.returncode,0,built.stderr)
            run = subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,f'Compact-tail case failed: {run.returncode} {run.stderr}')

    def test_roi_mask_interpolates_logits_before_argmax_with_exact_half_pixels(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "UPSAMPLE_SOURCE"
#include <vector>
#include <iostream>
static int logit(unsigned y,unsigned x,unsigned c,bool tie) {
  return tie?17:int((y*37+x*19+c*23+(x%3)*c*11)%121)-60;
}
static void axis(unsigned d,unsigned out,int& lo,int& hi,int& f) {
  int n=int((2*d+1)*30)-int(out), den=2*out;
  if(n<0) { lo=hi=0; f=0; }
  else if(n>=29*den) { lo=hi=29; f=0; }
  else { lo=n/den; hi=lo+1; f=n%den; }
}
int main() {
  using namespace esp_int8;
  const unsigned profiles[]={0,1,0,1,0};
  const unsigned sources[]={1,6,2,4,3};
  const unsigned counts[]={2,2,20,20,20};
  for(unsigned test=0;test<5;++test) {
    runtime_select_profile(profiles[test]);
    if(!runtime_select_io(sources[test],1)) return 1;
    runtime_io_t io; runtime_get_io(io);
    unsigned classes=counts[test]; bool tie=test==4;
    std::vector<axi_vec_t> output(io.mask_bytes/32+1); output.back()=0x9876;
    upsample_fused_begin();
    for(unsigned row=0;row<30;++row) {
      act_vec_t data[MAX_FM_W];
      for(unsigned x=0;x<30;++x) {
        data[x]=0;
        for(unsigned c=0;c<classes;++c) data[x].range(c*8+7,c*8)=logit(row,x,c,tie)&255;
      }
      if(!upsample_fused_consume_logits_row(output.data(),row,classes,data)) return 2;
    }
    if(s_mask_pack_fill || s_mask_word_index.to_uint()!=io.mask_bytes/32 || output.back()!=0x9876) return 3;
    for(unsigned y=0;y<io.mask_h;++y) for(unsigned x=0;x<io.mask_w;++x) {
      int y0,y1,fy,x0,x1,fx; axis(y,io.mask_h,y0,y1,fy); axis(x,io.mask_w,x0,x1,fx);
      int best=-1000000000,dx=2*io.mask_w,dy=2*io.mask_h; unsigned label=0;
      for(unsigned c=0;c<classes;++c) {
        int score=(logit(y0,x0,c,tie)*(dx-fx)+logit(y0,x1,c,tie)*fx)*(dy-fy)
                 +(logit(y1,x0,c,tie)*(dx-fx)+logit(y1,x1,c,tie)*fx)*fy;
        if(score>best) { best=score; label=c; }
      }
      unsigned b=y*io.mask_w+x;
      if(output[b/32].range((b%32)*8+7,(b%32)*8).to_uint()!=label) {
        std::cerr<<"roi mask "<<test<<" "<<y<<" "<<x; return 4;
      }
    }
    act_vec_t data[MAX_FM_W];
    if(upsample_fused_consume_logits_row(output.data(),0,classes,data)) return 5;
  }
  return 0;
}
'''.replace('UPSAMPLE_SOURCE', (root/'ESP_INT8_hls/src/upsample_unit.cpp').as_posix())
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'roi.cpp', Path(directory)/'roi.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, f'ROI mask case failed: {run.returncode} {run.stderr}')

    def test_input_dma_selects_gray_and_rgb_roi_pixels_without_requantization(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "npu_config.hpp"
#include "npu_types.hpp"
#include <vector>
#include <iostream>
namespace esp_int8 {
void frame_dma_load(const axi_vec_t*);
bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t&,u32_t,act_vec_t&);
bool on_chip_memory_write_fmbuf_abs_word(u8_t,u32_t,act_vec_t);
}
static unsigned value(unsigned i,unsigned seed) { return (i*17+i/191+seed*43)&255; }
int main() {
  using namespace esp_int8;
  for(unsigned p=0;p<8;++p) for(unsigned s=0;s<7;++s) {
    runtime_select_profile(p);
    if(!runtime_select_io(s,0)) continue;
    runtime_profile_t profile; runtime_get_profile(profile);
    runtime_io_t io; runtime_get_io(io);
    unsigned seed=p*7+s;
    std::vector<axi_vec_t> input(io.input_bytes/32);
    for(unsigned wi=0;wi<input.size();++wi) {
      input[wi]=0;
      for(unsigned lane=0;lane<32;++lane) input[wi].range(lane*8+7,lane*8)=value(wi*32+lane,seed);
    }
    on_chip_memory_write_fmbuf_abs_word(0,profile.input_bytes,0x1234);
    frame_dma_load(input.data());
    tensor_desc_t d; d.bank_id=0;
    for(unsigned wi=0;wi<profile.input_bytes/32;++wi) {
      act_vec_t word;
      if(!on_chip_memory_read_aligned_tensor_word(d,wi*32,word)) return 1;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned b=wi*32+lane, index=b;
        if(s) {
          unsigned pixel=b/profile.input_c, y=pixel/240, x=pixel%240;
          unsigned sy=io.roi_top+((2*y+1)*io.roi_h)/480;
          unsigned sx=io.roi_left+((2*x+1)*io.roi_w)/480;
          index=(sy*io.source_w+sx)*io.source_c+b%profile.input_c;
        }
        if(word.range(lane*8+7,lane*8).to_uint()!=value(index,seed)) {
          std::cerr<<"resize "<<p<<" "<<s<<" "<<b; return 2;
        }
      }
    }
    act_vec_t guard;
    if(!on_chip_memory_read_aligned_tensor_word(d,profile.input_bytes,guard) || guard!=0x1234) return 3;
  }
  runtime_select_profile(4); if(runtime_select_io(1,0)) return 4;
  frame_dma_load(nullptr);
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'input.cpp', Path(directory)/'input.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'),
                       str(root/'ESP_INT8_hls/src/frame_dma.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, f'Input DMA case failed: {run.returncode} {run.stderr}')

    def test_cpp_io_contract_matches_python_for_valid_and_invalid_selections(self):
        self.assertTrue(hasattr(compiler, 'render_runtime_hls_io'), 'runtime IO contract is missing')
        root = Path(__file__).resolve().parents[1]
        config = (root/'ESP_INT8_hls/include/npu_config.hpp').read_text(encoding='utf-8-sig')
        block = config[config.index('// BEGIN GENERATED RUNTIME IO'):]
        block = block[:block.index('// END GENERATED RUNTIME IO')+len('// END GENERATED RUNTIME IO')]
        self.assertEqual(block, compiler.render_runtime_hls_io())
        fields = tuple(geometry.runtime_io_fields(geometry.runtime_selection(0,2)))
        prints = ' << "," << '.join('io.'+field for field in fields)
        source = ('#include <iostream>\n#include "npu_config.hpp"\nint main() {'
                  'for(unsigned p=0;p<=8;++p) for(unsigned s=0;s<=7;++s) for(unsigned m=0;m<=2;++m) {'
                  'esp_int8::runtime_io_t io; bool ok=esp_int8::runtime_io_lookup(p,s,m,io);'
                  'std::cout<<p<<","<<s<<","<<m<<","<<ok<<","<<'+prints+'<<"\\n"; }'
                  'esp_int8::runtime_io_t io; '
                  'if(esp_int8::runtime_io_lookup(0xffffffffU,0,0,io)) return 1; return 0; }')
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'io.cpp', Path(directory)/'io.exe'
            path.write_text(source, encoding='ascii')
            built = subprocess.run([shutil.which('g++'), '-std=c++14',
                '-I'+str(root/'ESP_INT8_hls/include'), str(path), '-o', str(binary)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
        for line in run.stdout.splitlines():
            p,s,m,ok,*actual = map(int,line.split(','))
            try:
                selection = geometry.runtime_selection(p,2,src_mode=s,mask_mode=m)
            except ValueError:
                self.assertFalse(ok)
                self.assertEqual(actual,[0]*len(fields))
            else:
                self.assertTrue(ok)
                self.assertEqual(actual,list(geometry.runtime_io_fields(selection).values()))

    def test_scheduled_rgb_and_gray_windows_balance_tokens_from_bram(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "WIN_GEN_SOURCE"
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
namespace esp_int8 { bool on_chip_memory_write_fmbuf_abs_word(u8_t,u32_t,act_vec_t); }
static unsigned pattern(unsigned h,unsigned w,unsigned c) { return (h*29+w*17+c*7)&255; }
int main(int argc,char** argv) {
  using namespace esp_int8; runtime_select_profile(7);
  for(int test=1;test<argc;++test) {
    std::ifstream file(argv[test],std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
    if(bytes.size()!=128) return 1;
    auto u16=[&](unsigned i) { return bytes[i]+unsigned(bytes[i+1])*256; };
    window_sched_desc_t s;
    s.mode=bytes[0]; s.kernel=bytes[1]; s.stride=bytes[2]; s.dilation=bytes[3]; s.padding=bytes[4];
    s.cache_chunks=bytes[5]; s.cache_col_slots=bytes[6]; s.flags=bytes[7];
    s.in_c=u16(8); s.out_w=u16(10); s.k_tiles=u16(12); s.cmd_base=u16(14); s.cmd_count=u16(16);
    for(unsigned i=0;i<=MAX_K_TILE_COUNT;++i) s.kt_cmd_base[i]=u16(18+i*2);
    s.loader_class=bytes[100]; s.loader_request_cols=bytes[101]; s.loader_warmup_issues=bytes[102];
    s.loader_warmup_new_cols=bytes[103]; s.loader_steady_new_cols=bytes[104];
    s.loader_words_per_col=bytes[105]; s.loader_warmup_mask=bytes[106]; s.loader_steady_mask=bytes[107];
    for(unsigned i=0;i<10;++i) s.reserved[i]=u16(108+i*2);
    tensor_desc_t src; src.bank_id=0; src.elem_bytes=1; src.base_offset=FMBUF_URAM_BYTES;
    src.h=35; src.w=s.out_w.to_uint()*s.stride.to_uint(); src.c=s.in_c;
    src.reserved0=s.in_c.to_uint()|TENSOR_ROW_ALIGN32; src.reserved1=0;
    unsigned stride=tensor_row_stride_bytes(src).to_uint();
    for(unsigned h=0;h<src.h.to_uint();++h) for(unsigned wi=0;wi<stride/32;++wi) {
      act_vec_t word=0;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned byte=wi*32+lane;
        if(byte<src.w.to_uint()*src.c.to_uint())
          word.range(lane*8+7,lane*8)=pattern(h,byte/src.c.to_uint(),byte%src.c.to_uint());
      }
      if(!on_chip_memory_write_fmbuf_abs_word(0,src.base_offset+h*stride+wi*32,word)) return 2;
    }
    for(unsigned row: {0U,8U,17U}) {
      hls::stream<act_vec_t> a,b; volatile u8_t prof=0;
      scheduled_window_generator_row(src,s,a,b,row,prof);
      bool paired=(s.flags.to_uint()&1)!=0;
      unsigned issues=paired?(s.out_w.to_uint()+1)/2:s.out_w.to_uint();
      if(a.size()!=issues*s.k_tiles.to_uint() || b.size()!=(paired?a.size():0)) return 3;
      for(unsigned issue=0;issue<issues;++issue) for(unsigned kt=0;kt<s.k_tiles.to_uint();++kt) {
        act_vec_t words[2]={a.read(),0}; if(paired) words[1]=b.read();
        for(unsigned pixel=0;pixel<(paired?2U:1U);++pixel) for(unsigned lane=0;lane<32;++lane) {
          unsigned k=kt*32+lane, ow=issue*(paired?2:1)+pixel, expected=0;
          if(k<9*src.c.to_uint() && ow<s.out_w.to_uint()) {
            int h=int(row*s.stride.to_uint())-int(s.padding.to_uint())+int(k/(3*src.c.to_uint())*s.dilation.to_uint());
            int w=int(ow*s.stride.to_uint())-int(s.padding.to_uint())+int((k/src.c.to_uint()%3)*s.dilation.to_uint());
            if(h>=0 && h<int(src.h.to_uint()) && w>=0 && w<int(src.w.to_uint()))
              expected=pattern(h,w,k%src.c.to_uint());
          }
          if(words[pixel].range(lane*8+7,lane*8).to_uint()!=expected) {
            std::cerr<<"scheduled "<<test<<" "<<row<<" "<<ow<<" "<<k; return 4;
          }
        }
      }
    }
  }
  return 0;
}
'''.replace('WIN_GEN_SOURCE', (root/'ESP_INT8_hls/src/win_gen.cpp').as_posix())
        cases = ((19,30,2,1,True), (17,30,2,1,True), (131,60,2,1,False),
                 (129,60,2,1,False), (12,30,1,1,True), (12,30,1,16,True),
                 (25,30,1,16,False), (28,30,1,1,False), (64,30,1,1,False), (128,30,1,1,False))
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'scheduled.cpp', Path(directory)/'scheduled.exe'
            path.write_text(source, encoding='ascii')
            inputs = []
            for i,(channels,width,stride,dilation,paired) in enumerate(cases):
                sched,_ = compiler.make_window_pack_schedule(channels,3,source_w=width,
                    out_w=width//stride,stride=stride,dilation=dilation,padding=dilation,
                    flags=compiler.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 if paired else 0)
                candidate=Path(directory)/f's{i}.bin'
                candidate.write_bytes(compiler.pack_window_sched_desc(sched))
                inputs.append(str(candidate))
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary),*inputs], capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, f'Scheduled window case failed: {run.returncode} {run.stderr}')

    def test_top_init_and_prefix_zero_reject_stale_profiles_before_dma(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "TOP_CALL_HEADER"
#include "npu_uop.hpp"
#include "npu_ctrl.hpp"
#include <fstream>
#include <iterator>
#include <vector>
namespace esp_int8 { bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t&,u32_t,axi_vec_t&); }
int main(int argc,char** argv) {
  using namespace esp_int8;
  if(argc!=5) return 1;
  u8_t stage=0,pc=0,kind=0,win=0,sa=0,post=0; ap_uint<1> active=0;
  std::uint32_t abi=0,status=0,cap=0;
  auto call=[&](const axi_vec_t* input,const axi_vec_t* param,unsigned mode,unsigned profile,unsigned source=0,unsigned mask=0) {
    espnet_encoder_int8_core(input,nullptr,param,mode,75,stage,active,pc,kind,win,sa,post,profile,abi,status,source,mask,cap);
  };
  call(nullptr,nullptr,MODE_RUN,0);
  if(!((status>>8)&255) || (status&1) || stage!=PROF_STAGE_ERROR) return 2;
  call(nullptr,nullptr,MODE_INIT,8);
  if(abi!=RUNTIME_ABI_ID || (status&1) || param_dma_ready()) return 3;
  const unsigned profiles[5]={0,3,3,4,0};
  for(unsigned test=0;test<5;++test) {
    std::ifstream file(argv[test<4?test+1:1],std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
    if(bytes.empty()) return 4;
    std::vector<axi_vec_t> param(8192);
    for(unsigned i=0;i<bytes.size();++i) param[i/32].range((i%32)*8+7,(i%32)*8)=bytes[i];
    unsigned id=profiles[test];
    call(nullptr,param.data(),MODE_INIT,id);
    if((status&0xffff)!=1 || status>>16!=id || stage!=PROF_STAGE_IDLE) return 5;
    call(nullptr,nullptr,MODE_RUN,7);
    if(!((status>>8)&255) || stage!=PROF_STAGE_ERROR) return 6;
    call(nullptr,nullptr,MODE_RUN,id,7,0);
    if(!((status>>8)&255) || stage!=PROF_STAGE_ERROR) return 11;
    call(nullptr,nullptr,MODE_RUN,id,0,1);
    if(!((status>>8)&255) || stage!=PROF_STAGE_ERROR) return 12;
    runtime_profile_t p; runtime_get_profile(p);
    std::vector<axi_vec_t> input(p.input_bytes/32);
    for(unsigned i=0;i<input.size();++i) input[i]=i+test*1000;
    call(input.data(),nullptr,MODE_RUN,id);
    if((status&0xffff)!=1 || stage!=PROF_STAGE_IDLE || active!=0) return 7;
    tensor_desc_t d; d.bank_id=0; axi_vec_t read;
    if(!on_chip_memory_read_aligned_tensor_word(d,p.input_bytes-32,read) || read!=input.back()) return 8;
    param[0].range(31,0)=0;
    call(nullptr,param.data(),MODE_INIT,id);
    if((status&1) || param_dma_ready() || stage!=PROF_STAGE_ERROR) return 9;
    call(nullptr,nullptr,MODE_RUN,id);
    if((status&1) || stage!=PROF_STAGE_ERROR) return 10;
  }
  return 0;
}
'''.replace('TOP_CALL_HEADER', (root/'ESP_INT8_hls/tb/top_call.hpp').as_posix())
        # Prefix zero executes only input DMA and controller entry validation, not a network CSim.
        source = source.replace('namespace esp_int8 { bool on_chip',
                                'namespace esp_int8 { bool param_dma_ready(); bool on_chip')
        cases = ((0,2,'binary2_int8_h240w240_gray_r6_v4'),
                 (3,2,'binary2_int8_h256w512_r2_v4'),
                 (3,20,'cityscapes20_int8_h256w512_r2_v4'),
                 (4,2,'binary2_int8_h512w512_gray_finetuned_v4'))
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'top.cpp', Path(directory)/'top.exe'
            path.write_text(source, encoding='ascii')
            inputs = []
            for pid, classes, name in cases:
                candidate = Path(directory)/(name+'.bin')
                candidate.write_bytes(compiler.recompile_runtime_param(
                    (root/'hw_artifacts'/name/'PARAM.BIN').read_bytes(), geometry.runtime_selection(pid,classes)))
                inputs.append(str(candidate))
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-DESP_INT8_CSIM_MAX_UOP=0', '-ID:/Xilinx/2025.1/Vitis/include',
                       '-I'+str(root/'ESP_INT8_hls/include'), str(path),
                       *[str(p) for p in sorted((root/'ESP_INT8_hls/src').glob('*.cpp'))], '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary),*inputs], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, f'Top protocol case failed: {run.returncode} {run.stderr}')

    def test_vec_common_tail_preserves_b2_concat_through_1048_words(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "VEC_SOURCE"
#include <iostream>
namespace esp_int8 {
bool on_chip_memory_write_fmbuf_abs_word(u8_t,u32_t,act_vec_t);
bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t&,u32_t,act_vec_t&);
bool on_chip_memory_write_pool_bram_word(u8_t,u32_t,act_vec_t);
void reset_scratch_state();
bool param_dma_get_tensor_desc(u8_t,tensor_desc_t&) { return false; }
bool param_dma_get_fixed_exec_desc(u8_t,fixed_exec_desc_t&) { return false; }
bool param_dma_get_affine_qparam(u8_t,u8_t,aff_q_t& q) {
  for(unsigned c=0;c<32;++c) { q.mul[c]=1; q.bias[c]=0; q.shift[c]=0; q.reserved[c]=0; }
  return true;
}
}
static unsigned pattern(unsigned h,unsigned w,unsigned c) { return (h*13+w*11+c*7)&255; }
int main() {
  using namespace esp_int8;
  for(unsigned id:{0U,1U,7U}) {
    runtime_select_profile(id); runtime_profile_t p; runtime_get_profile(p); reset_scratch_state();
    tensor_desc_t src0={},src1={},src2={},dst={};
    src0.bank_id=0; src0.elem_bytes=1; src0.base_offset=0; src0.reserved1=0;
    src0.h=p.input_h/4; src0.w=p.input_w/4; src0.c=64; src0.reserved0=64;
    src1=src0; src1.base_offset=p.l20_base;
    src2=src0; src2.bank_id=0x81; src2.base_offset=0;
    src2.c=p.input_c; src2.reserved0=p.input_c|TENSOR_ROW_ALIGN32;
    dst=src0; dst.base_offset=0; dst.c=128+p.input_c; dst.reserved0=dst.c.to_uint()|TENSOR_ROW_ALIGN32;
    tensor_desc_t sources[3]={src0,src1,src2};
    for(unsigned which=0;which<3;++which) {
      const tensor_desc_t& d=sources[which];
      unsigned stride=tensor_row_stride_bytes(d).to_uint();
      for(unsigned h=0;h<d.h.to_uint();++h) for(unsigned wi=0;wi<stride/32;++wi) {
        act_vec_t word=0;
        for(unsigned lane=0;lane<32;++lane) {
          unsigned byte=wi*32+lane;
          if(byte<d.w.to_uint()*d.c.to_uint())
            word.range(lane*8+7,lane*8)=pattern(h,byte/d.c.to_uint(),which*64+byte%d.c.to_uint());
        }
        bool ok=which==2?on_chip_memory_write_pool_bram_word(d.bank_id,h*stride+wi*32,word)
                       :on_chip_memory_write_fmbuf_abs_word(d.bank_id,d.base_offset+h*stride+wi*32,word);
        if(!ok) return 1;
      }
    }
    fixed_exec_desc_t desc={}; desc.dst_tensor=TID_B2_ACT; desc.in_h=dst.h; desc.in_w=dst.w;
    desc.valid_c=dst.c; desc.flags=FIXED_FLAG_ROW_CONTIGUOUS_STORE; desc.param_id=0; desc.act_type=0;
    error_code_t err=run_fixed_affine_common(desc,src0,src1,src2,true,dst,true);
    if(err!=ERR_NONE) { std::cerr<<"vec error "<<id<<" "<<unsigned(err); return 2; }
    unsigned stride=tensor_row_stride_bytes(dst).to_uint();
    for(unsigned h=0;h<dst.h.to_uint();++h) for(unsigned wi=0;wi<stride/32;++wi) {
      act_vec_t word;
      if(!on_chip_memory_read_aligned_tensor_word(dst,h*stride+wi*32,word)) return 3;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned byte=wi*32+lane;
        unsigned expected=byte<dst.w.to_uint()*dst.c.to_uint()?pattern(h,byte/dst.c.to_uint(),byte%dst.c.to_uint()):0;
        if(word.range(lane*8+7,lane*8).to_uint()!=expected) {
          std::cerr<<"vec "<<id<<" "<<h<<" "<<byte<<" actual="
                   <<word.range(lane*8+7,lane*8).to_uint()<<" expected="<<expected; return 4;
        }
      }
    }
  }
  return 0;
}
'''.replace('VEC_SOURCE', (root/'ESP_INT8_hls/src/vec_alu_engine.cpp').as_posix())
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'vec.cpp', Path(directory)/'vec.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), *[str(root/'ESP_INT8_hls/src'/name) for name in
                                   ('memory.cpp', 'conv_store.cpp', 'scratch_mgr.cpp')], '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, f'Vec tail case failed: {run.returncode} {run.stderr}')

    def test_native_output_uses_active_rectangular_geometry_and_class_count(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "UPSAMPLE_SOURCE"
#include <vector>
#include <iostream>
static int logit(unsigned y,unsigned x,unsigned c) {
  return int((y*37+x*19+c*23+(x%3)*c*11)%121)-60;
}
static void axis(unsigned d,unsigned n,int& lo,int& hi,int& fraction) {
  int numerator=int(d*2)-7;
  if(numerator<0) { lo=hi=0; fraction=0; }
  else if(numerator>=int((n-1)*16)) { lo=hi=n-1; fraction=0; }
  else { lo=numerator/16; hi=lo+1; fraction=numerator%16; }
}
int main() {
  using namespace esp_int8;
  for(unsigned id=0;id<8;++id) for(unsigned classes:{2U,20U}) {
    runtime_select_profile(id); runtime_profile_t p; runtime_get_profile(p);
    std::vector<axi_vec_t> output(p.mask_bytes/32+1);
    output.back()=0x5a;
    upsample_fused_begin();
    for(unsigned row=0;row<3;++row) {
      act_vec_t data[MAX_FM_W];
      for(unsigned x=0;x<p.logits_w;++x) {
        data[x]=0;
        for(unsigned c=0;c<classes;++c) data[x].range(c*8+7,c*8)=logit(row,x,c)&255;
      }
      upsample_fused_consume_logits_row(output.data(),row,classes,data);
    }
    const unsigned bytes=s_mask_word_index.to_uint()*32;
    if(bytes!=20*p.input_w || s_mask_pack_fill) return 1;
    for(unsigned y=0;y<20;++y) for(unsigned x=0;x<p.input_w;++x) {
      int y0,y1,fy,x0,x1,fx; axis(y,p.logits_h,y0,y1,fy); axis(x,p.logits_w,x0,x1,fx);
      int best=-1000000000; unsigned expected=0;
      for(unsigned c=0;c<classes;++c) {
        int score=(logit(y0,x0,c)*(16-fx)+logit(y0,x1,c)*fx)*(16-fy)
                 +(logit(y1,x0,c)*(16-fx)+logit(y1,x1,c)*fx)*fy;
        if(score>best) { best=score; expected=c; }
      }
      unsigned byte=y*p.input_w+x;
      if(output[byte/32].range((byte%32)*8+7,(byte%32)*8).to_uint()!=expected) {
        std::cerr<<"mask "<<id<<" "<<classes<<" "<<y<<" "<<x; return 2;
      }
    }
    if(output.back()!=0x5a) return 3;
  }
  return 0;
}
'''.replace('UPSAMPLE_SOURCE', (root/'ESP_INT8_hls/src/upsample_unit.cpp').as_posix())
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'output.cpp', Path(directory)/'output.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, f'Native mask case failed: {run.returncode} {run.stderr}')

    def test_shared_pool_matches_signed_reference_with_padded_rows(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "npu_config.hpp"
#include "npu_types.hpp"
#include <vector>
#include <iostream>
namespace esp_int8 {
bool avgpool_unit_checked(const tensor_desc_t&,const tensor_desc_t&,const pool_q_t&,i8_t*);
bool on_chip_memory_write_fmbuf_abs_word(u8_t,u32_t,act_vec_t);
bool on_chip_memory_read_pool_bram_word(u8_t,u32_t,act_vec_t&);
bool on_chip_memory_write_pool_bram_word(u8_t,u32_t,act_vec_t);
}
int main() {
  using namespace esp_int8;
  for(unsigned id=0;id<8;++id) {
    runtime_select_profile(id); runtime_profile_t p; runtime_get_profile(p);
    tensor_desc_t src={}; src.bank_id=0; src.elem_bytes=1; src.base_offset=0; src.reserved1=0;
    src.h=6; src.w=p.input_w; src.c=p.input_c; src.reserved0=p.input_c;
    std::vector<int> input(src.h.to_uint()*src.w.to_uint()*p.input_c);
    for(unsigned i=0;i<input.size();++i) input[i]=int((i*23+i/17)%256)-128;
    for(unsigned wi=0;wi<(input.size()+31)/32;++wi) {
      act_vec_t word=0;
      for(unsigned lane=0;lane<32 && wi*32+lane<input.size();++lane)
        word.range(lane*8+7,lane*8)=input[wi*32+lane]&255;
      if(!on_chip_memory_write_fmbuf_abs_word(0,wi*32,word)) return 1;
    }
    for(unsigned level=0;level<2;++level) {
      tensor_desc_t dst={}; dst.bank_id=0x81; dst.elem_bytes=1; dst.base_offset=0; dst.reserved1=0;
      dst.h=(src.h.to_uint()+1)/2; dst.w=(src.w.to_uint()+1)/2;
      dst.c=p.input_c; dst.reserved0=p.input_c|TENSOR_ROW_ALIGN32;
      pool_q_t q={}; q.kernel=3; q.stride=2;
      q.same_scale=level==0; q.mult=3; q.shift=1; q.act_type=level;
      if(!avgpool_unit_checked(src,dst,q,nullptr)) return 2;
      std::vector<int> expected(dst.h.to_uint()*dst.w.to_uint()*p.input_c);
      for(unsigned h=0;h<dst.h.to_uint();++h) for(unsigned w=0;w<dst.w.to_uint();++w)
      for(unsigned c=0;c<p.input_c;++c) {
        int sum=0;
        for(int kh=0;kh<3;++kh) for(int kw=0;kw<3;++kw) {
          int ih=h*2+kh-1, iw=w*2+kw-1;
          if(ih>=0 && iw>=0 && ih<int(src.h.to_uint()) && iw<int(src.w.to_uint()))
            sum+=input[(ih*src.w.to_uint()+iw)*p.input_c+c];
        }
        int avg=sum>=0?(sum+4)/9:(sum-4)/9;
        if(level) { int product=avg*3; avg=product>=0?(product+1)/2:(product-1)/2; }
        avg=avg<-128?-128:avg>127?127:avg;
        if(level && avg<0) avg=0;
        expected[(h*dst.w.to_uint()+w)*p.input_c+c]=avg;
      }
      const unsigned stride=tensor_row_stride_bytes(dst).to_uint();
      for(unsigned h=0;h<dst.h.to_uint();++h) for(unsigned wi=0;wi<stride/32;++wi) {
        act_vec_t word;
        if(!on_chip_memory_read_pool_bram_word(dst.bank_id,h*stride+wi*32,word)) return 3;
        for(unsigned lane=0;lane<32;++lane) {
          unsigned b=wi*32+lane;
          unsigned want=b<dst.w.to_uint()*p.input_c?expected[h*dst.w.to_uint()*p.input_c+b]&255:0;
          if(word.range(lane*8+7,lane*8).to_uint()!=want) {
            std::cerr<<"pool "<<id<<" "<<level<<" "<<h<<" "<<b; return 4;
          }
        }
        if(!on_chip_memory_write_pool_bram_word(0x80,h*stride+wi*32,word)) return 5;
      }
      src=dst; src.bank_id=0x80; input=expected;
    }
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'pool.cpp', Path(directory)/'pool.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'),
                       str(root/'ESP_INT8_hls/src/avgpool_unit.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, f'AvgPool case failed: {run.returncode} {run.stderr}')

    def test_shared_first_window_and_rgb_channel_packing(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "WIN_GEN_SOURCE"
#include <iostream>
namespace esp_int8 { bool on_chip_memory_write_input_axi_word(u32_t,axi_vec_t); }
static unsigned pattern(unsigned index,unsigned seed) { return (index*13+index/239+seed*41)&255; }
int main() {
  using namespace esp_int8;
  for (unsigned run=0;run<10;++run) {
    const unsigned id=run<8?run:0;
    runtime_select_profile(id); runtime_profile_t p; runtime_get_profile(p);
    reset_window_state();
    tensor_desc_t src; src.bank_id=0; src.elem_bytes=1; src.base_offset=0; src.reserved1=0;
    src.h=p.input_h; src.w=p.input_w; src.c=p.input_c; src.reserved0=p.input_c;
    for(unsigned wi=0;wi<p.input_bytes/32;++wi) {
      axi_vec_t word=0;
      for(unsigned lane=0;lane<32;++lane) word.range(lane*8+7,lane*8)=pattern(wi*32+lane,run);
      if(!on_chip_memory_write_input_axi_word(wi,word)) return 1;
    }
    window_sched_desc_t sched;
    sched.mode=p.input_c==1?WIN_MODE_3X3_STAGED_C1:WIN_MODE_3X3_STAGED_C3;
    sched.out_w=p.input_w/2; sched.flags=1;
    const unsigned rows[]={0,1,2,p.input_h/2-1};
    for(unsigned row:rows) {
      hls::stream<act_vec_t> a,b; volatile u8_t prof=0;
      scheduled_window_generator_row(src,sched,a,b,row,prof);
      if(a.size()!=p.input_w/4 || b.size()!=p.input_w/4) return 2;
      for(unsigned issue=0;issue<p.input_w/4;++issue) {
        act_vec_t words[2]={a.read(),b.read()};
        for(unsigned pixel=0;pixel<2;++pixel) for(unsigned lane=0;lane<32;++lane) {
          unsigned expected=0;
          if(lane<9*p.input_c) {
            int h=int(row*2)+int(lane/(3*p.input_c))-1;
            int w=int(issue*4+pixel*2)+int((lane/p.input_c)%3)-1;
            if(h>=0 && h<int(p.input_h) && w>=0 && w<int(p.input_w))
              expected=pattern((h*p.input_w+w)*p.input_c+lane%p.input_c,run);
          }
          if(words[pixel].range(lane*8+7,lane*8).to_uint()!=expected) {
            std::cerr<<"first "<<id<<" "<<row<<" "<<issue<<" "<<pixel<<" "<<lane
                     <<" actual="<<words[pixel].range(lane*8+7,lane*8).to_uint()<<" expected="<<expected;
            return 3;
          }
        }
      }
    }
  }
  narrow_3x3_window_t narrow;
  wide_3x3_cache_t wide;
  const u8_t slots[3]={0,2,3};
  for(unsigned spatial=0;spatial<9;++spatial) {
    narrow.spatial[spatial]=0;
    for(unsigned lane=0;lane<19;++lane)
      narrow.spatial[spatial].range(lane*8+7,lane*8)=pattern(spatial*19+lane,0);
    for(unsigned chunk=0;chunk<5;++chunk) {
      act_vec_t word=0;
      for(unsigned lane=0;lane<32 && chunk*32+lane<131;++lane)
        word.range(lane*8+7,lane*8)=pattern(spatial*131+chunk*32+lane,0);
      wide.data[spatial/3][slots[spatial%3].to_uint()][chunk]=word;
    }
  }
  for(unsigned c: {19U,131U}) for(unsigned kt=0;kt<(9*c+31)/32;++kt) {
    act_vec_t word=0;
    if(c==19) build_staged_c19_word(narrow,kt,word);
    else build_staged_c131_word(wide,slots,kt,word);
    for(unsigned lane=0;lane<32;++lane) {
      unsigned k=kt*32+lane, expected=k<9*c?pattern(k,0):0;
      if(word.range(lane*8+7,lane*8).to_uint()!=expected) {
        std::cerr<<"packed "<<c<<" "<<kt<<" "<<lane; return 4;
      }
    }
  }
  return 0;
}
'''.replace('WIN_GEN_SOURCE', (root/'ESP_INT8_hls/src/win_gen.cpp').as_posix())
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'window.cpp', Path(directory)/'window.exe'
            path.write_text(source, encoding='ascii')
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, f'WinGen case failed: {run.returncode} {run.stderr}')

    def test_cpp_param_init_checks_runtime_layout_and_invalidates_failures(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "npu_config.hpp"
#include "npu_uop.hpp"
#include <fstream>
#include <iterator>
#include <vector>
namespace esp_int8 { void param_dma_init(const axi_vec_t*); void param_dma_invalidate(); bool param_dma_ready();
bool param_dma_get_affine_qparam(u8_t,u8_t,aff_q_t&); }
static unsigned u32le(const std::vector<unsigned char>& bytes,unsigned at) {
  unsigned value=0; for(unsigned j=0;j<4;++j) value|=unsigned(bytes[at+j])<<(8*j); return value;
}
static bool affine_words_match(const std::vector<unsigned char>& bytes) {
  using namespace esp_int8;
  unsigned cursor=u32le(bytes,76);
  const unsigned blocks[7]={1,2,2,5,4,4,8};
  aff_q_t q;
  for(unsigned id=0;id<7;++id) {
    for(unsigned block=0;block<blocks[id];++block,cursor+=320) {
      if(!param_dma_get_affine_qparam(id,block,q)) return false;
      for(unsigned lane=0;lane<32;++lane)
        if(q.mul[lane].range(31,0).to_uint()!=u32le(bytes,cursor+lane*4) ||
           q.bias[lane].range(31,0).to_uint()!=u32le(bytes,cursor+128+lane*4) ||
           q.shift[lane].to_uint()!=bytes[cursor+256+lane] ||
           q.reserved[lane].to_uint()!=bytes[cursor+288+lane]) return false;
    }
    if(param_dma_get_affine_qparam(id,blocks[id],q)) return false;
  }
  return !param_dma_get_affine_qparam(7,0,q) && !param_dma_get_affine_qparam(0,255,q);
}
int main(int argc,char** argv) {
  using namespace esp_int8;
  const unsigned profiles[4]={0,3,3,4};
  if (argc!=5) return 1;
  for (unsigned i=0;i<4;++i) {
    std::ifstream file(argv[i+1],std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
    if (bytes.size()<128) return 2;
    // Check every stored bit, including reserved bytes not used by inference.
    const unsigned affine=u32le(bytes,76);
    for(unsigned record=0;record<26;++record) for(unsigned lane=0;lane<32;++lane)
      bytes[affine+record*320+288+lane]=static_cast<unsigned char>(record*17+lane);
    std::vector<axi_vec_t> words(8192);
    for (unsigned b=0;b<bytes.size();++b) words[b/32].range((b%32)*8+7,(b%32)*8)=bytes[b];
    if (!runtime_select_profile(profiles[i])) return 3;
    param_dma_init(words.data());
    if (!param_dma_ready()) return 10+i;
    if (!affine_words_match(bytes)) return 60+i;
    unsigned table=0; for(unsigned j=0;j<4;++j) table|=unsigned(bytes[40+j])<<(8*j);
    const unsigned bad=table+2*16+4;
    words[bad/32].range((bad%32)*8+7,(bad%32)*8)=bytes[bad]^32;
    param_dma_init(words.data());
    if (param_dma_ready()) return 20+i;
    words[bad/32].range((bad%32)*8+7,(bad%32)*8)=bytes[bad];
    param_dma_init(words.data());
    if (!param_dma_ready()) return 30+i;
    if (!affine_words_match(bytes)) return 70+i;
    if (!runtime_select_profile(7)) return 4;
    param_dma_init(words.data());
    if (param_dma_ready()) return 40+i;
  }
  runtime_select_profile(8); param_dma_init(nullptr);
  if (param_dma_ready()) return 50;
  param_dma_invalidate();
  return param_dma_ready()?51:0;
}
'''
        cases = ((0,2,'binary2_int8_h240w240_gray_r6_v4'),
                 (3,2,'binary2_int8_h256w512_r2_v4'),
                 (3,20,'cityscapes20_int8_h256w512_r2_v4'),
                 (4,2,'binary2_int8_h512w512_gray_finetuned_v4'))
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'param.cpp', Path(directory)/'param.exe'
            path.write_text(source, encoding='ascii')
            inputs = []
            for pid, classes, name in cases:
                raw = (root/'hw_artifacts'/name/'PARAM.BIN').read_bytes()
                converted = compiler.recompile_runtime_param(raw, geometry.runtime_selection(pid, classes))
                candidate = Path(directory)/(name+'.bin')
                candidate.write_bytes(converted)
                inputs.append(str(candidate))
            command = [shutil.which('g++'), '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'),
                       str(root/'ESP_INT8_hls/src/param_dma.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary), *inputs], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, f'PARAM INIT case failed: {run.returncode} {run.stderr}')

    def test_fmbuf_addresses_and_aliases_survive_storage_resource_repair(self):
        root = Path(__file__).resolve().parents[1]
        memory = (root/'ESP_INT8_hls/src/memory.cpp').read_text(encoding='utf-8-sig')
        self.assertIn('s_fmbuf_uram[FMBUF_URAM_AXI_WORDS]', memory,
                      'restore one physical word bus after the banked routing regression')
        scratch = (root/'ESP_INT8_hls/src/scratch_mgr.cpp').read_text(encoding='utf-8-sig')
        self.assertNotIn('s_global_alias_epoch', scratch)
        source = r'''
#include "npu_config.hpp"
#include "npu_uop.hpp"
#include "npu_types.hpp"
#include "memory.cpp"
#include "scratch_mgr.cpp"
namespace esp_int8 {
bool param_dma_get_tensor_desc(u8_t id, tensor_desc_t& desc) {
  if (id.to_uint() >= MAX_TENSOR_DESC_COUNT) { desc={}; return false; }
  desc={}; desc.bank_id=BANK_FMEM0; desc.elem_bytes=1;
  desc.base_offset=4096+id.to_uint()*32; desc.h=1; desc.w=1; desc.c=32;
  return true;
}
}
static esp_int8::axi_vec_t pattern(unsigned word) {
  esp_int8::axi_vec_t value=0;
  for(unsigned lane=0;lane<8;++lane)
    value.range(lane*32+31,lane*32)=0x9e3779b9U*(1+word*8+lane);
  return value;
}
int main() {
  using namespace esp_int8;
  static_assert(FMBUF_URAM_AXI_WORDS==114688, "physical capacity must not shrink");
  static_assert(MAX_TENSOR_DESC_COUNT==19 && MAX_CONV_EXEC_DESC_COUNT==26, "descriptor bound");
  static_assert(MAX_AFFINE_PARAM_DESC_COUNT==7 && MAX_ADD_PARAM_DESC_COUNT==14 &&
                MAX_POOL_PARAM_DESC_COUNT==3 && MAX_ROW_CONSUMER_DESC_COUNT==26 &&
                MAX_FIXED_EXEC_DESC_COUNT==10 && MAX_EXEC_PLAN_COUNT==16, "finite ISA bound");
  if(!runtime_select_profile(7)) return 1;
  for(unsigned wi=0;wi<FMBUF_URAM_AXI_WORDS;++wi)
    if(!on_chip_memory_write_fmbuf_abs_word(BANK_FMEM0,wi*32,pattern(wi))) return 2;
  for(unsigned wi=0;wi<FMBUF_URAM_AXI_WORDS;++wi) {
    if(s_fmbuf_uram[wi]!=pattern(wi)) return 3;
    axi_vec_t got;
    if(!on_chip_memory_read_main_uram_word(wi%3,wi*32,got) || got!=pattern(wi)) return 4;
  }
  if(!on_chip_memory_write_fmbuf_abs_word(BANK_FMEM2,FMBUF_URAM_BYTES,pattern(123))) return 5;
  if(!on_chip_memory_write_fmbuf_abs_word(BANK_FMEM1,FMBUF_BYTES-32,pattern(456))) return 6;
  axi_vec_t got=1;
  if(on_chip_memory_read_main_uram_word(BANK_FMEM0,FMBUF_URAM_BYTES,got) || got!=0) return 7;
  if(write_fmbuf_uram_word(FMBUF_URAM_AXI_WORDS,0)) return 8;
  if(!on_chip_memory_read_fmbuf_abs_word(BANK_FMEM0,FMBUF_BYTES-32,got) || got!=pattern(456)) return 9;
  // Exercise adjacent words and the URAM/BRAM split in one span.
  const unsigned starts[]={3*32+17,FMBUF_URAM_BYTES-17};
  for(unsigned start:starts) {
    if(!on_chip_memory_read_packed_span(BANK_FMEM0,start,32,got)) return 10;
    const unsigned wi=start/32, shift=(start&31)*8;
    const axi_vec_t lo=pattern(wi);
    const axi_vec_t hi=(wi+1==FMBUF_URAM_AXI_WORDS)?pattern(123):pattern(wi+1);
    ap_uint<512> pair=hi; pair<<=256; pair|=lo;
    if(got!=static_cast<axi_vec_t>(pair>>shift)) return 11;
  }
  for(unsigned profile: {7U,0U,3U,4U,7U}) {
    if(!runtime_select_profile(profile)) return 12;
    reset_scratch_state();
    for(unsigned id=0;id<MAX_TENSOR_DESC_COUNT;++id) {
      tensor_desc_t base={}, desc={}; base.bank_id=BANK_FMEM0; base.elem_bytes=1;
      base.h=1; base.w=1; base.c=64; base.reserved0=64; base.base_offset=1024;
      if(!alias_global_tensor_to_slice(id,base,16,32) ||
         !resolve_tensor_read(id,desc) || desc.base_offset!=1024 || desc.reserved1!=16) return 13;
      if(!resolve_tensor_write(id,1,1,32,desc) || desc.base_offset!=4096+id*32 || desc.reserved1!=0) return 14;
      if(!alias_global_tensor_to_slice(id,base,0,32)) return 15;
    }
    reset_scratch_state();
    for(unsigned id=0;id<MAX_TENSOR_DESC_COUNT;++id) {
      tensor_desc_t desc={};
      if(!resolve_tensor_read(id,desc) || desc.base_offset!=4096+id*32 || desc.reserved1!=0) return 16;
    }
  }
  tensor_desc_t desc={};
  if(resolve_tensor_read(19,desc) || alias_global_tensor_to_slice(19,desc,0,0)) return 17;
  return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='esp_storage_') as directory:
            path, exe = Path(directory)/'storage.cpp', Path(directory)/'storage.exe'
            path.write_text(source, encoding='ascii')
            command = ['D:/mingw64/bin/g++.exe', '-std=c++14', '-O2', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       '-I'+str(root/'ESP_INT8_hls/src'), str(path), '-o', str(exe)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, f'storage/alias regression: {run.returncode} {run.stderr}')

    def test_memory_and_store_use_active_layout_and_full_width_cursors(self):
        root = Path(__file__).resolve().parents[1]
        source = r'''
#include "npu_config.hpp"
#include "npu_types.hpp"
namespace esp_int8 {
bool on_chip_memory_write_input_axi_word(u32_t, axi_vec_t);
bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t&, u32_t, axi_vec_t&);
bool on_chip_memory_write_pool_bram_word(u8_t, u32_t, axi_vec_t);
bool on_chip_memory_read_pool_bram_word(u8_t, u32_t, axi_vec_t&);
bool conv_store_word_span_plan_ok(u8_t, u64_t, u32_t, u16_t);
bool on_chip_memory_write_fmbuf_abs_word(u8_t,u32_t,act_vec_t);
void reset_scratch_state();
bool b2_src1_row_base(const tensor_desc_t&,u16_t,u32_t&);
bool backup_b2_src1_rows_before_write(const tensor_desc_t&,const tensor_desc_t&,u16_t);
bool param_dma_get_tensor_desc(u8_t,tensor_desc_t&) { return false; }
}
int main() {
  using namespace esp_int8;
  for (unsigned id=0; id<8; ++id) {
    if (!runtime_select_profile(id) || !runtime_profile_ready()) return 1;
    runtime_profile_t p; runtime_get_profile(p);
    axi_vec_t value=0x123400U+id, read=0;
    if (!on_chip_memory_write_input_axi_word(p.input_bytes/32-1,value)) return 2;
    if (on_chip_memory_write_input_axi_word(p.input_bytes/32,value)) return 3;
    if (!on_chip_memory_write_pool_bram_word(0x81,p.pool2_bytes-32,value)) return 4;
    if (on_chip_memory_write_pool_bram_word(0x81,p.pool2_bytes,value)) return 5;
    if (!on_chip_memory_read_pool_bram_word(0x81,p.pool2_bytes-32,read) || read!=value) return 6;
    tensor_desc_t absolute; absolute.bank_id=0;
    if (!on_chip_memory_read_aligned_tensor_word(absolute,p.pool1_base+p.pool2_bytes-32,read) || read!=value) return 7;
    if (!on_chip_memory_write_pool_bram_word(0x80,p.pool1_bytes-32,value)) return 8;
    if (on_chip_memory_write_pool_bram_word(0x80,p.pool1_bytes,value)) return 9;
    if (!conv_store_word_span_plan_ok(0x81,p.pool2_bytes-32,32,1)) return 10;
    if (conv_store_word_span_plan_ok(0x81,p.pool2_bytes-32,32,2)) return 15;
    reset_scratch_state();
    tensor_desc_t residual, destination;
    residual.bank_id=0; residual.elem_bytes=1; residual.base_offset=p.l20_base;
    residual.h=p.input_h/4; residual.w=p.input_w/4; residual.c=64; residual.reserved0=64;
    destination=residual; destination.base_offset=0;
    destination.c=128+p.input_c; destination.reserved0=destination.c.to_uint()|TENSOR_ROW_ALIGN32;
    const unsigned stride=tensor_row_stride_bytes(residual).to_uint();
    const unsigned dest_stride=tensor_row_stride_bytes(destination).to_uint();
    for (unsigned row=0;row<residual.h.to_uint();++row) {
      for (unsigned wi=0;wi<stride/32;++wi) {
        if (!on_chip_memory_write_fmbuf_abs_word(0,p.l20_base+row*stride+wi*32,1+row*1024+wi)) return 16;
      }
    }
    for (int row=residual.h.to_uint()-1;row>=0;--row) {
      u32_t base;
      if (!b2_src1_row_base(residual,row,base)) return 17;
      for (unsigned wi=0;wi<stride/32;++wi) {
        if (!on_chip_memory_read_aligned_tensor_word(residual,base+wi*32,read) || read!=1+row*1024+wi) return 18;
      }
      if (!backup_b2_src1_rows_before_write(residual,destination,row)) return 19;
      for (unsigned wi=0;wi<dest_stride/32;++wi) {
        if (!on_chip_memory_write_fmbuf_abs_word(0,row*dest_stride+wi*32,0)) return 20;
      }
    }
  }
  if (!conv_store_word_span_plan_ok(0,0,32,1048)) return 11;
  if (conv_store_word_span_plan_ok(0,FMBUF_BYTES-1048*32+32,32,1048)) return 12;
  if (runtime_select_profile(8) || runtime_profile_ready()) return 13;
  runtime_profile_t p; runtime_get_profile(p);
  if (p.input_bytes || on_chip_memory_write_input_axi_word(0,0)) return 14;
  return 0;
}
'''
        compiler_path = shutil.which('g++')
        self.assertIsNotNone(compiler_path)
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'memory.cpp', Path(directory)/'memory.exe'
            path.write_text(source, encoding='ascii')
            command = [compiler_path, '-std=c++14', '-Wno-unknown-pragmas',
                       '-ID:/Xilinx/2025.1/Vitis/include', '-I'+str(root/'ESP_INT8_hls/include'),
                       str(path), str(root/'ESP_INT8_hls/src/memory.cpp'),
                       str(root/'ESP_INT8_hls/src/conv_store.cpp'),
                       str(root/'ESP_INT8_hls/src/scratch_mgr.cpp'), '-o', str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, f'C++ memory/store case failed: {run.returncode} {run.stderr}')

    def test_cpp_profile_table_matches_python_and_rejects_bad_ids(self):
        self.assertTrue(hasattr(compiler, 'render_runtime_hls_profiles'), 'C++ runtime contract is missing')
        root = Path(__file__).resolve().parents[1]
        header = root/'ESP_INT8_hls/include/npu_config.hpp'
        text = header.read_text(encoding='utf-8-sig')
        start = text.index('// BEGIN GENERATED RUNTIME PROFILES')
        end = text.index('// END GENERATED RUNTIME PROFILES')+len('// END GENERATED RUNTIME PROFILES')
        self.assertEqual(text[start:end], compiler.render_runtime_hls_profiles())
        fields = tuple(compiler.runtime_profile_fields(geometry.RUNTIME_PROFILES[0]))
        prints = ' << "," << '.join('p.'+field for field in fields)
        source = ('#include <iostream>\n#include "'+header.as_posix()+'"\n'
                  'int main() { for (unsigned id=0; id<8; ++id) { '
                  'esp_int8::runtime_profile_t p; '
                  'if (!esp_int8::runtime_profile_lookup(id,p)) return 1; '
                  'std::cout << '+prints+' << "\\n"; } '
                  'esp_int8::runtime_profile_t p; '
                  'if (esp_int8::runtime_profile_lookup(8,p) || p.input_bytes) return 2; '
                  'if (esp_int8::runtime_profile_lookup(0xffffffffU,p)) return 3; return 0; }')
        compiler_path = shutil.which('g++')
        self.assertIsNotNone(compiler_path, 'host C++ compiler is required for the POD contract test')
        with tempfile.TemporaryDirectory() as directory:
            path, binary = Path(directory)/'profile.cpp', Path(directory)/'profile.exe'
            path.write_text(source, encoding='ascii')
            built = subprocess.run([compiler_path, '-std=c++14', str(path), '-o', str(binary)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = [tuple(map(int, row.split(','))) for row in run.stdout.splitlines()]
        expected = [tuple(compiler.runtime_profile_fields(p).values()) for p in geometry.RUNTIME_PROFILES]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
