cmake_minimum_required(VERSION 3.13)
project(bbp CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#arch must be changed to match your GPU
set(CMAKE_CUDA_FLAGS "-arch=sm_75")

find_package(Boost 1.70.0 EXACT COMPONENTS system chrono)
IF(Boost_FOUND)
	set(OPENSSL_USE_STATIC_LIBS TRUE)
	find_package(OpenSSL REQUIRED)

	add_executable(bbp main.cpp bbpLauncher.cpp digitData.cpp restClientDelegator.cpp progressData.cpp kernel.cu)
	target_link_libraries(bbp Boost::system Boost::chrono OpenSSL::SSL OpenSSL::Crypto)
ELSE()
	add_definitions(-DNO_BOOST)
	add_executable(bbp main.cpp bbpLauncher.cpp digitData.cpp progressData.cpp kernel.cu)
ENDIF()