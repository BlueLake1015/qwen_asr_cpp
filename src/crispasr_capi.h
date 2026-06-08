// crispasr_capi.h — declarations for the subset of CrispASR's C-ABI we link
// against (libcrispasr.so) for the per-segment "aligned streaming" path.
//
// These symbols are exported (extern "C") by the engine library. The session +
// vad functions are stock CrispASR; the `crispasr_paligner_*` trio is the
// persistent forced-aligner shim added in
// third_party/CrispASR/src/crispasr_aligner.cpp so the aligner GGUF loads once
// and is reused across segments (no per-segment reload).
#pragma once

#include <cstdint>

extern "C" {

// ---- ASR session (load model once, reuse) ----
struct crispasr_session;
struct crispasr_session_result;

crispasr_session* crispasr_session_open(const char* model_path, int n_threads);
void              crispasr_session_close(crispasr_session* s);

crispasr_session_result* crispasr_session_transcribe(crispasr_session* s,
                                                     const float* pcm, int n_samples);
int         crispasr_session_result_n_segments(crispasr_session_result* r);
const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i);
void        crispasr_session_result_free(crispasr_session_result* r);

// ---- VAD segmentation (returns malloc'd [start_s, end_s] pairs in seconds) ----
int  crispasr_vad_slices(const char* vad_model_path, const float* pcm, int n_samples,
                         int sample_rate, float threshold, int min_speech_ms,
                         int min_silence_ms, int speech_pad_ms, float max_chunk_duration_s,
                         int n_threads, float** out_spans);
void crispasr_vad_free(float* spans);

// ---- persistent forced aligner (our shim; load once, reuse) ----
struct crispasr_paligner;
crispasr_paligner* crispasr_paligner_open(const char* model_path, int n_threads);
// Align n_words (char* ptrs) to samples; writes centisecond timings (offset by
// t_offset_cs) into out_t0_cs/out_t1_cs. Returns 0 on success.
int  crispasr_paligner_align(crispasr_paligner* a, const char** words, int n_words,
                             const float* samples, int n_samples, int64_t t_offset_cs,
                             int64_t* out_t0_cs, int64_t* out_t1_cs);
void crispasr_paligner_close(crispasr_paligner* a);

}  // extern "C"
