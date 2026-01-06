set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__GLST_DEBUG__")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D__GLST_DEBUG__")

add_library(GLST
  ${GLST_SOURCE_DIR}/src/cubature.cu
  ${GLST_SOURCE_DIR}/src/cuda_container.cu
  ${GLST_SOURCE_DIR}/src/cuda_utils.cu
  ${GLST_SOURCE_DIR}/src/device_vector.cu
  ${GLST_SOURCE_DIR}/src/glst_force.cu
)
set_target_properties(GLST PROPERTIES
  CUDA_ARCHITECTURES "80"
)
# set_target_properties(GLST PROPERTIES
#   CUDA_ARCHITECTURES "60;61;62;70;72;75;80;86;87;89;90;90a"
# )
# set_target_properties(GLST PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
# )
# set_target_properties(GLST PROPERTIES
#   COMPILE_FLAGS "-rdc=true"
# )
# set_target_properties(GLST PROPERTIES
#   LINK_FLAGS "-lcudadevrt"
# )
target_include_directories(GLST PUBLIC ${GLST_SOURCE_DIR}/include)

target_link_libraries(GLST
  CUDA::cudart
  # CUDA::cublas
  CUDA::cuda_driver
)
