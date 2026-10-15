/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vlm_engine.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{
//! Check if the current CUDA device is an integrated GPU (iGPU / Jetson Orin)
bool isIntegratedGPU()
{
    int device{-1};
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return false;
    }
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess)
    {
        return false;
    }
    return integrated == 1;
}

//! Get current GPU free and total memory
std::pair<size_t, size_t> queryGpuMemory()
{
    size_t freeMem{0}, totalMem{0};
    cudaMemGetInfo(&freeMem, &totalMem);
    return {freeMem, totalMem};
}

//! Read process RSS in bytes on Linux
size_t getProcessRssBytes()
{
#ifndef _WIN32
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        // ru_maxrss is in kilobytes on Linux, convert to bytes
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
#endif
    return 0;
}

//! Self-contained memory monitor built with -fPIC (matches official MemoryMonitor behavior without non-PIC link error)
class InProcessMemoryMonitor
{
public:
    InProcessMemoryMonitor() = default;
    ~InProcessMemoryMonitor()
    {
        stop();
    }

    void start()
    {
        mIsIGPU = isIntegratedGPU();
        mPeakGpuMemory = 0;
        mActive = true;

        if (!mIsIGPU)
        {
            auto [freeMem, totalMem] = queryGpuMemory();
            mBaselineGpuFreeMemory = freeMem;
            mWorker = std::thread([this]() {
                while (mActive.load())
                {
                    auto [currentFreeMem, totalMem] = queryGpuMemory();
                    size_t gpuMemoryUsed = (currentFreeMem < mBaselineGpuFreeMemory)
                        ? (mBaselineGpuFreeMemory - currentFreeMem)
                        : 0;
                    if (mPeakGpuMemory < gpuMemoryUsed)
                    {
                        mPeakGpuMemory = gpuMemoryUsed;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
    }

    void stop()
    {
        mActive = false;
        if (mWorker.joinable())
        {
            mWorker.join();
        }
    }

    size_t getPeakUnifiedMemory() const
    {
        return getProcessRssBytes();
    }

    size_t getPeakGpuMemory() const
    {
        return mPeakGpuMemory.load();
    }

    bool isIGPU() const
    {
        return mIsIGPU;
    }

private:
    std::atomic<bool> mActive{false};
    std::atomic<size_t> mPeakGpuMemory{0};
    size_t mBaselineGpuFreeMemory{0};
    bool mIsIGPU{false};
    std::thread mWorker;
};

struct VLMEngineContext
{
    std::unique_ptr<void, DlDeleter> pluginHandles{nullptr};
    std::unique_ptr<rt::LLMInferenceRuntime> runtime{nullptr};
    cudaStream_t stream{nullptr};
    std::mutex mtx;

    double coldStartMs{0.0};
    double graphCaptureMs{0.0};
    double lastInferenceMs{0.0};
    int lastGeneratedTokens{0};
    int lastPromptTokens{0};
    int lastImageTokens{0};
    double lastVisMs{0.0};
    double lastPrefillMs{0.0};
    double lastGenMs{0.0};
    double lastTtftMs{0.0};
    double lastTpotMs{0.0};
    double peakMemoryMb{0.0};
    std::string lastOutputBuffer;
};
} // namespace

extern "C" {

void* vlm_create(const char* engineDir, const char* multimodalEngineDir)
{
    if (!engineDir || std::strlen(engineDir) == 0)
    {
        std::cerr << "[VLMEngine] [Error] engineDir is required." << std::endl;
        return nullptr;
    }

    // Auto-detect EDGELLM_PLUGIN_PATH if not explicitly set
    if (std::getenv("EDGELLM_PLUGIN_PATH") == nullptr)
    {
        std::vector<std::string> candPaths = {
            "libNvInfer_edgellm_plugin.so",
            "../libNvInfer_edgellm_plugin.so",
            "../../libNvInfer_edgellm_plugin.so",
            "build/libNvInfer_edgellm_plugin.so"
        };
#ifndef _WIN32
        try
        {
            if (std::filesystem::exists("/proc/self/exe"))
            {
                auto exePath = std::filesystem::canonical("/proc/self/exe");
                auto binDir = exePath.parent_path();
                candPaths.push_back((binDir / "libNvInfer_edgellm_plugin.so").string());
                candPaths.push_back((binDir / "../libNvInfer_edgellm_plugin.so").string());
                candPaths.push_back((binDir / "../../libNvInfer_edgellm_plugin.so").string());
            }
        }
        catch (...)
        {
        }
#endif
        for (auto const& p : candPaths)
        {
            if (std::filesystem::exists(p))
            {
                std::cout << "[VLMEngine] 自动定位插件库: " << p << std::endl;
                std::cout.flush();
#ifndef _WIN32
                setenv("EDGELLM_PLUGIN_PATH", p.c_str(), 1);
#endif
                break;
            }
        }
    }

    auto ctx = std::make_unique<VLMEngineContext>();
    ctx->pluginHandles = loadEdgellmPluginLib();

    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));

    auto t0 = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, std::string> loraMap;
    std::string mmDir = (multimodalEngineDir && std::strlen(multimodalEngineDir) > 0) ? multimodalEngineDir : "";

    std::filesystem::path normEngineDir(engineDir);
    if (!std::filesystem::exists(normEngineDir))
    {
        std::cerr << "[VLMEngine] [Fatal] engineDir does not exist: " << engineDir << std::endl;
        return nullptr;
    }

    try
    {
        ctx->runtime = std::make_unique<rt::LLMInferenceRuntime>(
            normEngineDir.string(), mmDir, loraMap, ctx->stream);
    }
    catch (std::exception const& e)
    {
        std::cerr << "[VLMEngine] [Fatal] Failed to create LLMInferenceRuntime: " << e.what() << std::endl;
        if (ctx->stream)
        {
            cudaStreamDestroy(ctx->stream);
        }
        return nullptr;
    }

    auto tRuntime = std::chrono::high_resolution_clock::now();
    double runtimeLoadMs = std::chrono::duration<double, std::milli>(tRuntime - t0).count();

    // Check if CUDA Graph capture is disabled via environment variable (default enabled)
    bool enableGraph = true;
    const char* envNoGraph = std::getenv("VLM_DISABLE_CUDA_GRAPH");
    if (envNoGraph && (std::strcmp(envNoGraph, "1") == 0 || std::strcmp(envNoGraph, "true") == 0))
    {
        enableGraph = false;
        std::cout << "[VLMEngine] CUDA Graph capture explicitly disabled via VLM_DISABLE_CUDA_GRAPH." << std::endl;
    }

    if (enableGraph)
    {
        // Capture decoding CUDA graph (matching llm_inference standard process)
        auto tGraph0 = std::chrono::high_resolution_clock::now();
        bool graphCaptured = ctx->runtime->captureDecodingCUDAGraph(ctx->stream);
        auto tGraph1 = std::chrono::high_resolution_clock::now();
        ctx->graphCaptureMs = std::chrono::duration<double, std::milli>(tGraph1 - tGraph0).count();

        if (!graphCaptured)
        {
            cudaGetLastError(); // Clear sticky error status from failed stream capture
            std::cerr << "[VLMEngine] [Warning] Failed to capture CUDA graph for decoding, proceeding with normal execution." << std::endl;
        }
    }

    auto tTotal = std::chrono::high_resolution_clock::now();
    ctx->coldStartMs = std::chrono::duration<double, std::milli>(tTotal - t0).count();

    std::cout << "[VLMEngine] [Ready] VLM in-process engine instantiated. "
              << "Runtime load: " << std::fixed << std::setprecision(1) << runtimeLoadMs << " ms, "
              << "CUDA Graph capture: " << ctx->graphCaptureMs << " ms, "
              << "Total cold start: " << ctx->coldStartMs << " ms (" << (ctx->coldStartMs / 1000.0) << " s)." << std::endl;
    std::cout.flush();

    return ctx.release();
}

const char* vlm_generate(void* handle, const char* prompt, const char* imagePath, int maxTokens)
{
    if (!handle)
    {
        return "";
    }
    auto* ctx = static_cast<VLMEngineContext*>(handle);
    std::lock_guard<std::mutex> lock(ctx->mtx);

    if (!ctx->runtime)
    {
        return "";
    }

    // Reset previous generation metrics
    ctx->lastPromptTokens = 0;
    ctx->lastGeneratedTokens = 0;
    ctx->lastImageTokens = 0;
    ctx->lastVisMs = 0.0;
    ctx->lastPrefillMs = 0.0;
    ctx->lastGenMs = 0.0;
    ctx->lastTtftMs = 0.0;
    ctx->lastTpotMs = 0.0;
    ctx->peakMemoryMb = 0.0;
    ctx->lastOutputBuffer.clear();

    rt::LLMGenerationRequest req;
    req.temperature = 0.0f;
    req.topP = 1.0f;
    req.topK = 1;
    req.maxGenerateLength = (maxTokens > 0) ? maxTokens : 128;
    req.applyChatTemplate = true;
    req.addGenerationPrompt = true;
    req.enableThinking = false;

    std::vector<rt::Message> chatMessages;
    std::vector<rt::imageUtils::ImageData> imageBuffers;

    rt::Message msg;
    msg.role = "user";

    // Load image if path provided and exists
    if (imagePath && std::strlen(imagePath) > 0 && std::filesystem::exists(imagePath))
    {
        auto imgData = rt::imageUtils::loadImageFromFile(imagePath);
        if (imgData.buffer != nullptr)
        {
            rt::Message::MessageContent imgContent;
            imgContent.type = "image";
            imgContent.content = imagePath;
            msg.contents.push_back(std::move(imgContent));
            imageBuffers.push_back(std::move(imgData));
        }
    }

    rt::Message::MessageContent textContent;
    textContent.type = "text";
    textContent.content = (prompt && std::strlen(prompt) > 0) ? prompt : "";
    msg.contents.push_back(std::move(textContent));

    chatMessages.push_back(std::move(msg));
    req.requests = {rt::LLMGenerationRequest::Request{chatMessages, imageBuffers, {}}};

    // Baseline counters to track per-request tokens accurately
    int64_t prevReused = ctx->runtime->getPrefillMetrics().reusedTokens;
    int64_t prevComputed = ctx->runtime->getPrefillMetrics().computedTokens;
    int64_t prevImageTokens = ctx->runtime->getMultimodalMetrics().totalImageTokens;

    // Reset timer data for this specific run
    gTimer.reset();

    // Enable TensorRT-Edge-LLM profiling and start memory monitor
    setProfilingEnabled(true);
    InProcessMemoryMonitor memoryMonitor;
    memoryMonitor.start();

    auto tInf0 = std::chrono::high_resolution_clock::now();
    rt::LLMGenerationResponse resp;
    bool status = ctx->runtime->handleRequest(req, resp, ctx->stream);
    auto tInf1 = std::chrono::high_resolution_clock::now();

    // Stop profiling and memory monitor immediately
    setProfilingEnabled(false);
    memoryMonitor.stop();

    ctx->lastInferenceMs = std::chrono::duration<double, std::milli>(tInf1 - tInf0).count();

    if (status && !resp.outputTexts.empty())
    {
        ctx->lastOutputBuffer = resp.outputTexts[0];
    }
    else
    {
        ctx->lastOutputBuffer.clear();
    }

    if (!resp.outputIds.empty())
    {
        ctx->lastGeneratedTokens = static_cast<int>(resp.outputIds[0].size());
    }
    else
    {
        ctx->lastGeneratedTokens = 0;
    }

    // Token delta calculations
    int64_t deltaReused = ctx->runtime->getPrefillMetrics().reusedTokens - prevReused;
    int64_t deltaComputed = ctx->runtime->getPrefillMetrics().computedTokens - prevComputed;
    int64_t deltaImageTokens = ctx->runtime->getMultimodalMetrics().totalImageTokens - prevImageTokens;

    ctx->lastPromptTokens = static_cast<int>(deltaReused + deltaComputed);
    ctx->lastImageTokens = static_cast<int>(deltaImageTokens);

    // Extract GPU microsecond timing from gTimer
    auto prefillTiming = gTimer.getTimingData(metrics::StageNames::kLLM_PREFILL);
    auto genTiming = gTimer.getTimingData(metrics::StageNames::kLLM_GENERATION);
    auto visTiming = gTimer.getTimingData(metrics::StageNames::kVISION_ENCODER);
    if (!visTiming)
    {
        visTiming = gTimer.getTimingData(metrics::StageNames::kMULTIMODAL_PROCESSING);
    }

    if (visTiming && visTiming->getTotalGpuTimeMs() > 0.0f)
    {
        ctx->lastVisMs = static_cast<double>(visTiming->getTotalGpuTimeMs());
    }
    if (prefillTiming && prefillTiming->getTotalGpuTimeMs() > 0.0f)
    {
        ctx->lastPrefillMs = static_cast<double>(prefillTiming->getTotalGpuTimeMs());
    }
    if (genTiming && genTiming->getTotalGpuTimeMs() > 0.0f)
    {
        ctx->lastGenMs = static_cast<double>(genTiming->getTotalGpuTimeMs());
    }

    // TTFT = Vision Encoder + LLM Prefill
    if (ctx->lastVisMs > 0.0 || ctx->lastPrefillMs > 0.0)
    {
        ctx->lastTtftMs = ctx->lastVisMs + ctx->lastPrefillMs;
    }

    // TPOT: Time Per Output Token
    if (ctx->lastGeneratedTokens > 1 && ctx->lastGenMs > 0.0)
    {
        ctx->lastTpotMs = ctx->lastGenMs / static_cast<double>(ctx->lastGeneratedTokens - 1);
    }
    else if (ctx->lastGeneratedTokens > 0 && ctx->lastGenMs > 0.0)
    {
        ctx->lastTpotMs = ctx->lastGenMs / static_cast<double>(ctx->lastGeneratedTokens);
    }

    if (memoryMonitor.isIGPU())
    {
        size_t peakBytes = memoryMonitor.getPeakUnifiedMemory();
        ctx->peakMemoryMb = static_cast<double>(peakBytes) / (1024.0 * 1024.0);
    }
    else
    {
        size_t peakBytes = memoryMonitor.getPeakGpuMemory();
        ctx->peakMemoryMb = static_cast<double>(peakBytes) / (1024.0 * 1024.0);
    }

    return ctx->lastOutputBuffer.c_str();
}

double vlm_get_cold_start_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->coldStartMs;
}

double vlm_get_graph_capture_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->graphCaptureMs;
}

double vlm_get_last_inference_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastInferenceMs;
}

int vlm_get_last_generated_tokens(void* handle)
{
    if (!handle) return 0;
    return static_cast<VLMEngineContext*>(handle)->lastGeneratedTokens;
}

double vlm_get_last_vis_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastVisMs;
}

double vlm_get_last_prefill_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastPrefillMs;
}

double vlm_get_last_gen_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastGenMs;
}

double vlm_get_last_ttft_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastTtftMs;
}

double vlm_get_last_tpot_ms(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->lastTpotMs;
}

double vlm_get_peak_memory_mb(void* handle)
{
    if (!handle) return 0.0;
    return static_cast<VLMEngineContext*>(handle)->peakMemoryMb;
}

int vlm_get_last_prompt_tokens(void* handle)
{
    if (!handle) return 0;
    return static_cast<VLMEngineContext*>(handle)->lastPromptTokens;
}

int vlm_get_last_image_tokens(void* handle)
{
    if (!handle) return 0;
    return static_cast<VLMEngineContext*>(handle)->lastImageTokens;
}

void vlm_destroy(void* handle)
{
    if (!handle)
    {
        return;
    }
    auto* ctx = static_cast<VLMEngineContext*>(handle);
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->runtime.reset();
        if (ctx->stream)
        {
            cudaStreamDestroy(ctx->stream);
            ctx->stream = nullptr;
        }
        ctx->pluginHandles.reset();
    }
    delete ctx;
    std::cout << "[VLMEngine] [Clean] Engine destroyed, 100% Unified Memory reclaimed." << std::endl;
    std::cout.flush();
}

} // extern "C"
