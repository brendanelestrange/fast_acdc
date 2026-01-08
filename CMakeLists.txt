cmake_minimum_required(VERSION 3.7)
project(acdc_proj)

# --- MacOS OpenMP Fix ---
if(APPLE)
    # Check if Homebrew libomp is installed and add it to paths
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/opt/homebrew/opt/libomp/include")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/opt/libomp/include")
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp") # For Intel Macs
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
    endif()
endif()
# ------------------------

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# --- Optimization Flags ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
    )
elseif(MSVC)
    add_compile_options(/O2 /fp:fast)
endif()
# --------------------------

# Add the 'libraries' folder to include path
include_directories(libraries)

add_executable(acdc 
    acdc.cpp 
    libraries/lapjv.cpp 
)

# Link OpenMP and Eigen
if(OpenMP_CXX_FOUND)
    target_link_libraries(acdc 
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
    )
endif()