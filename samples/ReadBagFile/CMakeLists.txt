
set(CATKIN_ENABLE_TESTING=0)

find_package(catkin REQUIRED COMPONENTS cv_bridge rosbag roscpp sensor_msgs)

include_directories(include
        ${catkin_INCLUDE_DIRS}
        )

add_executable(sample-read-bag read_bag.cpp)
target_link_libraries(sample-read-bag PUBLIC ${catkin_LIBRARIES} PRIVATE x)

add_custom_command(
        TARGET sample-read-bag  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/samples/ReadBagFile/UAV0_params_thermal.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/UAV0_params_thermal.yaml)
