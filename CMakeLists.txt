cmake_minimum_required(VERSION 3.10)
project(ImuAHRS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})

# Library
add_library(ImuAHRS
    src/MadgwickAHRS.cpp
    src/ImuData.cpp
)

target_link_libraries(ImuAHRS PUBLIC ${EIGEN3_LIBRARIES})

# Test executable
add_executable(test_ahrs
    test/test_ahrs.cpp
)

target_link_libraries(test_ahrs PRIVATE ImuAHRS)

# Comparison test executable
add_executable(compare_ahrs
    test/compare_ahrs.cpp
)

target_link_libraries(compare_ahrs PRIVATE ImuAHRS)

# Install
install(TARGETS ImuAHRS LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
