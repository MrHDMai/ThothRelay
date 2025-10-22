cmake_minimum_required(VERSION 3.10)
project(ThothOSRMBackend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to OSRM include and lib dirs (adjust if needed)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Add your source files
add_executable(thoth_osrm
    src/main.cpp
    src/osrm_wrapper.cpp
    src/route_handler.cpp
)

# Link OSRM libraries
target_link_libraries(thoth_osrm
    osrm
    osrm_extract
    osrm_contract
    osrm_store
    osrm_util
    boost_system
    boost_filesystem
    pthread
)
