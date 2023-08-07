cmake_minimum_required(VERSION 3.13)
project(test_dailyvecm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

add_executable(test_dailyvecm DailyVECM.cpp DailyVECM.h VECM.h VECM.cpp)


target_link_libraries(test_dailyvecm
    ${CMAKE_SOURCE_DIR}/../../../../TransferEntropy/libTransferEntropyLib.a
    ${CMAKE_SOURCE_DIR}/../../../../TransferEntropy/libckdtree.a)