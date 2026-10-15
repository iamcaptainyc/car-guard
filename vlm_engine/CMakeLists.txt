# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME vlm_engine)

add_library(${TARGET_NAME} SHARED
    vlm_engine.cpp
)

target_compile_definitions(${TARGET_NAME} PRIVATE VLM_ENGINE_EXPORTS)

# Match llm_inference release optimizations and symbols
target_compile_options(${TARGET_NAME} PRIVATE -O3 -DNDEBUG -fvisibility=default)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMMON_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/cpp
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        edgellmCore
        commonLibraryExt
)

add_cross_build_link_options(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
