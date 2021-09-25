cmake_minimum_required(VERSION 3.2)

project(playback LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(SOURCE_FILES 
    src/main.cpp 
    src/ukf.cpp 
    src/tools.cpp 
    src/render/render.cpp)


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


add_executable (ukf_highway ${SOURCE_FILES})
target_link_libraries (ukf_highway ${PCL_LIBRARIES})




