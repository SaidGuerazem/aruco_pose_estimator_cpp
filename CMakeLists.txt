cmake_minimum_required(VERSION 3.8)
project(aruco_pose_estimator_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d aruco)

add_executable(aruco_pose_estimator src/aruco_pose_estimator.cpp)

ament_target_dependencies(
  aruco_pose_estimator
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

target_link_libraries(aruco_pose_estimator ${OpenCV_LIBS})

install(TARGETS
  aruco_pose_estimator
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
