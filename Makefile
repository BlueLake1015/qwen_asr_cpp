# qwen_asr_cpp — video/audio -> transcription with Qwen3-ASR (C++/ggml).
#
# Two parts, both driven by this Makefile:
#   1. The CrispASR inference engine (vendored submodule, pure C++/ggml, no
#      Python) — built via its own CMake to produce the `crispasr` binary.
#   2. Our `qwen-asr` orchestrator — decodes media with the external ffmpeg
#      command, then runs the engine. Tiny; plain g++.
#
# ffmpeg is an external *runtime* dependency (media decode). cmake is used only
# to compile the vendored engine.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
JOBS     ?= $(shell nproc 2>/dev/null || echo 4)

# GPU backend for the engine. Default: CUDA (all inference runs on the GPU).
#   make GPU=cuda   (default)   make GPU=off   (CPU-only fallback)
# CUDA 12.0's nvcc rejects gcc>=13, so we pin g++-12 as the CUDA host compiler.
GPU            ?= cuda
CUDA_HOST_CXX  ?= g++-12
CUDA_ARCH      ?= 89          # NVIDIA Ada (RTX 40xx). Override for other GPUs.
ifeq ($(GPU),cuda)
CMAKE_GPU := -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_CXX) -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
else
CMAKE_GPU :=
endif

ROOT        := $(CURDIR)
ENGINE_DIR  := third_party/CrispASR
ENGINE_BUILD:= $(ENGINE_DIR)/build
ENGINE_BIN  := $(ENGINE_BUILD)/bin/crispasr
ENGINE_LIB  := $(ENGINE_BUILD)/src/libcrispasr.so

BUILD  := build
OBJDIR := $(BUILD)/obj
BIN    := $(BUILD)/qwen-asr

APP_OBJS := $(OBJDIR)/audio_decode.o $(OBJDIR)/main.o

# Bake the engine binary path and the default models dir into our wrapper
# (overridable at runtime: --crispasr/$QASR_CRISPASR, --models-dir/$QASR_MODELS_DIR).
DEFS := -DCRISPASR_BIN=\"$(ROOT)/$(ENGINE_BIN)\" -DMODELS_DIR=\"$(ROOT)/models\"

# Link the C-ABI (libcrispasr.so + ggml) for the --aligned per-segment path.
# rpath points at the in-tree libs so the binary runs without LD_LIBRARY_PATH.
LIB_SRC  := $(ENGINE_BUILD)/src
GGML_SRC := $(ENGINE_BUILD)/ggml/src
LDLIBS   := -L$(LIB_SRC) -L$(GGML_SRC) -lcrispasr -lggml -lggml-base \
            -Wl,-rpath,$(ROOT)/$(LIB_SRC) -Wl,-rpath,$(ROOT)/$(GGML_SRC)

.PHONY: all engine clean clean-all distclean help submodules patch

all: $(BIN)

help:
	@echo "Targets:"
	@echo "  make            - build the engine + our app -> $(BIN)"
	@echo "  make engine     - build only the CrispASR engine binary"
	@echo "  make submodules - fetch the CrispASR submodule"
	@echo "  make patch      - apply the engine patch for --aligned (explicit; see patches/)"
	@echo "  make clean      - remove our build/ (keeps the compiled engine)"
	@echo "  make clean-all  - also remove the engine's CMake build"
	@echo ""
	@echo "Get models (no Python):  scripts/get_model.sh"
	@echo "Run: $(BIN) input.mp4 -o out.srt --aligner models/qwen3-forced-aligner-0.6b-q4_k.gguf"

submodules:
	git submodule update --init --recursive --depth 1
	@echo ">> NOTE: the --aligned path needs the engine patch. Apply it with: make patch"

# Apply our local engine patch(es): the persistent-aligner shim for --aligned.
# Explicit, idempotent. Run after a fresh 'make submodules' or a submodule bump
# (which resets the submodule and drops the shim). NOT run automatically.
patch:
	bash patches/apply.sh

$(ENGINE_DIR)/CMakeLists.txt:
	@echo ">> engine submodule missing; run 'make submodules'" >&2; exit 1

# ---- CrispASR engine (CMake) ----
engine: $(ENGINE_BIN)

$(ENGINE_BIN) $(ENGINE_LIB) &: | $(ENGINE_DIR)/CMakeLists.txt
	@echo ">> building CrispASR engine + library (GPU=$(GPU); heavy step)"
	cmake -S $(ENGINE_DIR) -B $(ENGINE_BUILD) -DCMAKE_BUILD_TYPE=Release $(CMAKE_GPU)
	cmake --build $(ENGINE_BUILD) -j $(JOBS) --target crispasr-cli crispasr-lib
	@test -x $(ENGINE_BIN) && echo ">> engine built: $(ENGINE_BIN)"

# ---- our app ----
$(OBJDIR)/%.o: src/%.cpp src/audio_decode.h src/crispasr_capi.h | $(ENGINE_BIN)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEFS) -c $< -o $@

$(BIN): $(APP_OBJS) $(ENGINE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(APP_OBJS) $(LDLIBS) -o $@
	@echo ">> built $@"

clean:
	rm -rf $(BUILD)

clean-all: clean
	rm -rf $(ENGINE_BUILD)

distclean: clean-all
