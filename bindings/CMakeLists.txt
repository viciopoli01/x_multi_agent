cmake_minimum_required(VERSION 2.18...3.16)

# Find Python installation and its NumPy support. Execute before pybind11 to prime CMake caches.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

add_subdirectory(external/pybind11 EXCLUDE_FROM_ALL)

PYBIND11_ADD_MODULE(
        x_bind
        SHARED
        src/x.cpp
        external/pybind11cv/ndarray_converter.cpp)

target_compile_features(x_bind PRIVATE cxx_std_17)

target_link_libraries(x_bind PRIVATE x Python3::Module Python3::NumPy)

install(TARGETS x_bind DESTINATION ${Python3_SITEARCH})


# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
# file(COPY tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tests)

# add_test(
#         NAME test-python-bindings
#         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_all_tests.py
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
#
# set_tests_properties(test-python-bindings PROPERTIES ENVIRONMENT PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR})

message(WARNING "To make the bindings work, add '/opt/x/lib/cmake/x' to your LD_LIBRARY_PATH:\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/x/lib/cmake/x")
