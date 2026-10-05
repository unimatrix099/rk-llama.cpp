# Finding OpenMP
find_package(OpenMP REQUIRED)

# Adding backend files
ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
    rknpu2-quantization.cpp
    rknpu2-calibration.cpp
    rknpu2-configuration.cpp
    rknpu2-native-layout.c
)

# Linking rknpu2 header files
target_include_directories(ggml-rknpu2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Linking rknpu2 library
target_link_libraries(ggml-rknpu2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so
)

target_link_options(ggml-rknpu2 PUBLIC
    "-Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/libs"
)

# Adding OpenMP in build
if(OpenMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    target_compile_options(ggml-rknpu2 PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(ggml-rknpu2 PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Pin function and loop alignment: the dequant kernel's hot loop measured
# ~10% slower when unrelated code elsewhere in the library shifted it onto a
# bad alignment (NPU prefill loop 3, #5); 64-byte alignment makes the speed
# independent of layout.
target_compile_options(ggml-rknpu2 PRIVATE -falign-functions=64 -falign-loops=64)
