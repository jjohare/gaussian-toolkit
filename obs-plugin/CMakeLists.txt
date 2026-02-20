cmake_minimum_required(VERSION 3.20)
project(obs-lichtfeld-source VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

find_package(LibObs QUIET)
if(NOT LibObs_FOUND)
    if(NOT DEFINED OBS_SOURCE_DIR)
        message(STATUS "OBS not found. Set OBS_SOURCE_DIR to build the plugin.")
        return()
    endif()
    set(CMAKE_MODULE_PATH ${OBS_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
    find_package(LibObs REQUIRED)
endif()

add_library(obs-lichtfeld-source MODULE
    plugin.cpp
    shm_source.cpp
    shm_source.hpp
)

target_include_directories(obs-lichtfeld-source PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/visualizer
)

target_link_libraries(obs-lichtfeld-source PRIVATE
    ${LIBOBS_LIBRARIES}
    rt
)

target_include_directories(obs-lichtfeld-source PRIVATE
    ${LIBOBS_INCLUDE_DIRS}
)

set_target_properties(obs-lichtfeld-source PROPERTIES
    PREFIX ""
)
