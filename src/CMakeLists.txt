add_library(surround_system "")
target_link_libraries(surround_system ${THIRD_LIBRARIES})
target_compile_options(surround_system PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)

add_subdirectory(gui)
add_subdirectory(common)
add_subdirectory(SV)