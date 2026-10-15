/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(_WIN32)
#if defined(VLM_ENGINE_EXPORTS)
#define VLM_API __declspec(dllexport)
#else
#define VLM_API __declspec(dllimport)
#endif
#else
#define VLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize and load VLM TensorRT engines (single cold-start load into unified memory).
 * @param engineDir Path to the LLM TensorRT engine directory (e.g. models/engines/llm).
 * @param multimodalEngineDir Path to the visual TensorRT engine directory (e.g. models/engines/visual).
 * @return Opaque handle pointer to the VLMEngine instance, or NULL on failure.
 */
VLM_API void* vlm_create(const char* engineDir, const char* multimodalEngineDir);

/**
 * @brief Perform in-process VLM multimodal inference.
 * @param handle Opaque handle returned by vlm_create.
 * @param prompt Text prompt string (e.g. instructions and telemetry context).
 * @param imagePath Optional path to an image file (JPEG/PNG), or NULL/empty string for text-only.
 * @param maxTokens Maximum new tokens to generate (e.g. 128).
 * @return Pointer to a null-terminated UTF-8 string containing model response.
 *         The returned pointer is valid until the next call to vlm_generate or vlm_destroy on this handle.
 */
VLM_API const char* vlm_generate(void* handle, const char* prompt, const char* imagePath, int maxTokens);

/**
 * @brief Get the total cold start load and initialization time in milliseconds.
 */
VLM_API double vlm_get_cold_start_ms(void* handle);

/**
 * @brief Get the decoding CUDA Graph pre-capture time in milliseconds.
 */
VLM_API double vlm_get_graph_capture_ms(void* handle);

/**
 * @brief Get the pure inference time in milliseconds of the most recent vlm_generate call.
 */
VLM_API double vlm_get_last_inference_ms(void* handle);

/**
 * @brief Get the number of generated tokens of the most recent vlm_generate call.
 */
VLM_API int vlm_get_last_generated_tokens(void* handle);

/**
 * @brief Destroy the VLM instance and release 100% of Unified Memory / VRAM back to OS.
 */
VLM_API void vlm_destroy(void* handle);

#ifdef __cplusplus
} // extern "C"

#include <string>
#include <memory>

namespace guard
{

/**
 * @brief Modern C++ RAII wrapper around the C API of libvlm_engine.so.
 *        Can be directly included and compiled in any native C++ application.
 */
class VLMEngine
{
public:
    explicit VLMEngine(const std::string& engineDir, const std::string& multimodalEngineDir = "")
        : handle_(vlm_create(engineDir.c_str(), multimodalEngineDir.empty() ? nullptr : multimodalEngineDir.c_str()))
    {
    }

    ~VLMEngine()
    {
        release();
    }

    // Disable copy semantics, enable move semantics
    VLMEngine(const VLMEngine&) = delete;
    VLMEngine& operator=(const VLMEngine&) = delete;

    VLMEngine(VLMEngine&& other) noexcept
        : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    VLMEngine& operator=(VLMEngine&& other) noexcept
    {
        if (this != &other)
        {
            release();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool isValid() const noexcept
    {
        return handle_ != nullptr;
    }

    std::string infer(const std::string& prompt, const std::string& imagePath = "", int maxTokens = 128)
    {
        if (!handle_)
        {
            return "";
        }
        const char* res = vlm_generate(
            handle_, prompt.c_str(), imagePath.empty() ? nullptr : imagePath.c_str(), maxTokens);
        return res ? std::string(res) : "";
    }

    double getColdStartMs() const noexcept
    {
        return handle_ ? vlm_get_cold_start_ms(handle_) : 0.0;
    }

    double getGraphCaptureMs() const noexcept
    {
        return handle_ ? vlm_get_graph_capture_ms(handle_) : 0.0;
    }

    double getLastInferenceMs() const noexcept
    {
        return handle_ ? vlm_get_last_inference_ms(handle_) : 0.0;
    }

    int getLastGeneratedTokens() const noexcept
    {
        return handle_ ? vlm_get_last_generated_tokens(handle_) : 0;
    }

    void release()
    {
        if (handle_)
        {
            vlm_destroy(handle_);
            handle_ = nullptr;
        }
    }

private:
    void* handle_{nullptr};
};

} // namespace guard
#endif
