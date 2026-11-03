// Encode 44100 Hz mono reference audio to ordinary C-order [frames,10] codes.
#include "dac-encoder.h"
#include "ssl-encoder.h"
#include "audio-decode.h"
#include "model-paths.h"
#include "npy.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,"usage: %s <dac-encoder.gguf> <ssl-teacher.gguf> --audio <ref.mp3|pcm.npy> <out.npy> [--gpu|--cpu] [--dump <directory>]\n",argv[0]);
        return 1;
    }
    std::string audio_path,output,dir;
    bool gpu=false;
    for (int i=3;i<argc;++i) {
        const std::string a=argv[i];
        if (a=="--gpu") gpu=true;
        else if (a=="--cpu") gpu=false;
        else if (a=="--audio" && i+1<argc) audio_path=argv[++i];
        else if (a=="--dump" && i+1<argc) dir=argv[++i];
        else if (!a.empty() && a[0]=='-') { fprintf(stderr,"unknown/incomplete flag %s\n",a.c_str()); return 1; }
        else if (output.empty()) output=a;
        else { fprintf(stderr,"unexpected argument %s\n",a.c_str()); return 1; }
    }
    if (audio_path.empty() || output.empty()) { fprintf(stderr,"--audio and output path are required\n"); return 1; }
    std::vector<float> audio;
    if (audio_path.size()>=4 && audio_path.substr(audio_path.size()-4)==".npy") {
        std::vector<int64_t> shape;
        if (!npy::load_f32(audio_path,audio,shape) || !(shape.size()==1 || (shape.size()==2 && shape[0]==1))) { fprintf(stderr,"PCM NPY must be F32 [samples] or [1,samples] at 44100 Hz\n"); return 1; }
    } else audio=audio_decode_file(audio_path.c_str(),44100,DAC_ENCODER_MAX_SECONDS);
    if (audio.empty() || audio.size()>(size_t)DAC_ENCODER_MAX_SECONDS*44100) { fprintf(stderr,"reference must contain 1..%d mono samples\n",DAC_ENCODER_MAX_SECONDS*44100); return 1; }
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir,ec);
        if (ec) { fprintf(stderr,"cannot create dump directory: %s\n",ec.message().c_str()); return 1; }
        if (!npy::save_f32(dir+"/audio.npy",audio.data(),{(int64_t)audio.size()})) return 1;
    }
    dac_encoder_model encoder;
    ssl_model teacher;
    if (!dac_encoder_load(encoder,argv[1],gpu)) return 1;
    if (!ssl_load(teacher,argv[2],gpu)) { dac_encoder_free(encoder); return 1; }
    dac_encoder_dump dump;
    if (!dir.empty()) dump=[&](const std::string & name,const float * data,int T,int C) {
        return npy::save_f32(dir+"/"+name+".npy",data,{T,C});
    };
    std::vector<int32_t> codes;
    const auto start=std::chrono::steady_clock::now();
    bool ok=dac_encode_diagnostic(encoder,teacher,audio.data(),(int)audio.size(),codes,dump);
    if (ok) ok=mp_save_reference_codes(output,codes);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    ssl_free(teacher); dac_encoder_free(encoder);
    if (!ok) { fprintf(stderr,"reference encoding/output failed\n"); return 1; }
    fprintf(stderr,"encoded %zu samples -> %zu frames x 10 ordinary codes in %.3fs\n",audio.size(),codes.size()/10,seconds);
    return 0;
}
