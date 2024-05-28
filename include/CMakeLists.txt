add_library(common_sample STATIC common/filesys_util.cpp)
target_include_directories(common_sample PUBLIC common)

add_subdirectory(gui)
add_subdirectory(SV)
add_subdirectory(Utils)