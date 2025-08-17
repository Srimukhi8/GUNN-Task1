cmake_minimum_required(VERSION 3.10)
project(GraphGNN LANGUAGES CXX)

# Choose C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All source files
set(SOURCE_FILES
    GATL.cpp
    GCNL.cpp
    GCNTest.cpp
    Graph.cpp
    GraphReader.cpp
    GraphSage.cpp
    output.cpp
    output_main.cpp
)

# Build the executable
add_executable(graph_app ${SOURCE_FILES})

# Make headers in this folder visible
target_include_directories(graph_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# After building graph_app, copy graph_data.txt into the build folder
add_custom_command(TARGET graph_app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/graph_data.txt"
        "${CMAKE_BINARY_DIR}/graph_data.txt"
    COMMENT "Copying graph_data.txt to build directory"
)
