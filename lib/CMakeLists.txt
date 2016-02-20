cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14") # Enable C++14 features
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall") # Turn on warnings

set(SOURCES_NN
	NeuralNetwork.cpp
	)

add_library(libSNN ${SOURCES_NN} )
# target_link_libraries(libSNN ...)
