set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__GLST_DEBUG__")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D__GLST_DEBUG__")

add_library(GLST
  ${GLST_SOURCE_DIR}/src/cubature.cu
  ${GLST_SOURCE_DIR}/src/cuda_container.cu
  ${GLST_SOURCE_DIR}/src/cuda_utils.cu
  ${GLST_SOURCE_DIR}/src/device_comm.cu
  ${GLST_SOURCE_DIR}/src/device_vector.cu
  ${GLST_SOURCE_DIR}/src/glst_force.cu
  ${GLST_SOURCE_DIR}/src/string_utils.cpp
)

set_target_properties(GLST PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${GLST_CUDA_ARCHITECTURES}"
  # CUDA_SEPARABLE_COMPILATION ON
  # COMPILE_FLAGS "-rdc=true"
  # LINK_FLAGS "-lcudadevrt"
)

target_include_directories(GLST PUBLIC ${GLST_SOURCE_DIR}/include)

target_link_libraries(GLST
  CUDA::cudart
  # CUDA::cublas
  CUDA::cuda_driver
  nccl
)
