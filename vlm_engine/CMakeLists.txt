# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME vlm_engine)

add_library(${TARGET_NAME} SHARED
    vlm_engine.cpp
)

target_compile_definitions(${TARGET_NAME} PRIVATE VLM_ENGINE_EXPORTS)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/cpp
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        edgellmCore
        exampleUtils
        commonLibraryExt
)

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
