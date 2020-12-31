# This code is released under the
# Apache License Version 2.0 http://www.apache.org/licenses/.
#
# Copyright (c) 2012 Louis Dionne
#

cmake_minimum_required(VERSION 2.8)
project(Codecs)

include_directories(include)

add_library(Codecs_lib STATIC generate/SIMDMasks.cpp
                              generate/varintGBTables.cpp
                              generate/varintGUTables.cpp)
