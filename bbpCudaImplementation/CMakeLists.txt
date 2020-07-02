cmake_minimum_required(VERSION 3.13)
project(bbp CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)
find_package(Boost 1.70.0 EXACT REQUIRED COMPONENTS system chrono)

add_executable(bbp main.cpp bbpLauncher.cpp digitData.cpp restClientDelegator.cpp progressData.cpp kernel.cu)
target_link_libraries(bbp Boost::system Boost::chrono OpenSSL::SSL OpenSSL::Crypto)