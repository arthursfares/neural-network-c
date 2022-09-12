cmake_minimum_required(VERSION 3.0.0)
project(neural_network_c VERSION 0.1.0)

include_directories(great_old_headers)
add_executable(neural_network_c
    src/main.c
    great_old_headers/eldritch_arrays.h
    great_old_headers/neural_network.h
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
