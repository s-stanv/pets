cmake_minimum_required(VERSION 3.18)
project(cuda_optimizers LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(cuda_optimizers
  src/main.cu
  src/optimizers.cu
  src/utils.cu
)

set_target_properties(cuda_optimizers PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

if (MSVC)
  target_compile_options(cuda_optimizers PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
  )
else()
  target_compile_options(cuda_optimizers PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic>
  )
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
