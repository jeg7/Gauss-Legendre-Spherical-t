cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(GLST
  VERSION 1.0
  LANGUAGES CXX CUDA
)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Generate compile_commands.json by default for tools such as clangd.
  # Useres can disable this with: -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF
  #
  # If an old build directory has an empty cached value, treat that as
  # "not configured" and reset it to the project default.
  if(DEFINED CMAKE_EXPORT_COMPILE_COMMANDS AND
      "${CMAKE_EXPORT_COMPILE_COMMANDS}" STREQUAL "")
    unset(CMAKE_EXPORT_COMPILE_COMMANDS CACHE)
  endif()
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
    "Export compile_commands.json for language servers")

  # Specify language standards
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(GLST_CUDA_ARCHITECTURES "60;70;80;90" CACHE STRING
    "CUDA architectures to build for"
  )

  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${GLST_SOURCE_DIR}/bin
  )

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${GLST_SOURCE_DIR}/lib
  )

  # Compiler flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wno-deprecated-gpu-targets")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g3 -G -O0")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# NCCL
if(NOT DEFINED ENV{NCCL_ROOT})
  message(FATAL_ERROR "NCCL_ROOT was not defined in your environment")
endif()
set(NCCL_LIB $ENV{NCCL_ROOT})
set(NCCL_INCLUDE $ENV{NCCL_ROOT})
string(APPEND NCCL_LIB "/lib/libnccl.so")
string(APPEND NCCL_INCLUDE "/include")
message("-- NCCL lib: ${NCCL_LIB}")
message("-- NCCL include: ${NCCL_INCLUDE}")
add_library(nccl SHARED IMPORTED)
set_target_properties(nccl PROPERTIES
  IMPORTED_LOCATION ${NCCL_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE}
)

add_subdirectory(${GLST_SOURCE_DIR}/src)
add_subdirectory(${GLST_SOURCE_DIR}/utils)

enable_testing()
add_subdirectory(${GLST_SOURCE_DIR}/test)
