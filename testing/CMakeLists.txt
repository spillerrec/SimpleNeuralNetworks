cmake_minimum_required(VERSION 3.0)
project(Overmix)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14") # Enable C++14 features
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall") # Turn on warnings


set(SOURCE_TESTING
	main.cpp
	)

add_executable(NNTest ${SOURCE_TESTING})

include_directories ("${PROJECT_SOURCE_DIR}/../lib")
target_link_libraries(NNTest libSNN)