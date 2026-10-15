/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vlm_engine.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{
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

    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

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

    // Capture decoding CUDA graph (matching llm_inference standard process)
    auto tGraph0 = std::chrono::high_resolution_clock::now();
    bool graphCaptured = ctx->runtime->captureDecodingCUDAGraph(ctx->stream);
    auto tGraph1 = std::chrono::high_resolution_clock::now();
    ctx->graphCaptureMs = std::chrono::duration<double, std::milli>(tGraph1 - tGraph0).count();

    if (!graphCaptured)
    {
        std::cerr << "[VLMEngine] [Warning] Failed to capture CUDA graph for decoding, proceeding with normal execution." << std::endl;
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

    auto tInf0 = std::chrono::high_resolution_clock::now();
    rt::LLMGenerationResponse resp;
    bool status = ctx->runtime->handleRequest(req, resp, ctx->stream);
    auto tInf1 = std::chrono::high_resolution_clock::now();

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
