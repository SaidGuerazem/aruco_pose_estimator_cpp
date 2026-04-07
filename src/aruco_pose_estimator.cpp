#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class ArucoPoseEstimator : public rclcpp::Node
{
public:
  ArucoPoseEstimator()
  : Node("aruco_pose_estimator"),
    marker_length_(1.0)
  {
    // ------------------------------------------------------------
    // Camera topics
    // ------------------------------------------------------------
    camera_topics_ = {
      "/cam_1/color/image_raw",
      "/cam_2/color/image_raw"
    };

    // ------------------------------------------------------------
    // Extrinsics: ^D T_C  (Camera frame -> Drone frame)
    //
    // If p_C is a point in camera coordinates:
    //   p_D = ^D T_C * p_C
    // ------------------------------------------------------------
    camera_to_drone_tf_["/cam_1/color/image_raw"] = cv::Matx44d(
       0.0,  0.0,  1.0,   0.0,
      -1.0,  0.0,  0.0,  -0.12037,
       0.0, -1.0,  0.0,  -0.11435,
       0.0,  0.0,  0.0,   1.0
    );

    camera_to_drone_tf_["/cam_2/color/image_raw"] = cv::Matx44d(
       0.0,  0.70710678118,  0.70710678118,   0.0,
       1.0,  0.0,            0.0,            -0.12037,
       0.0,  0.70710678118, -0.70710678118, -0.11435,
       0.0,  0.0,            0.0,             1.0
    );

    // ------------------------------------------------------------
    // Camera intrinsics
    // ------------------------------------------------------------
    calibrations_["/cam_1/color/image_raw"] = {
      (cv::Mat_<double>(3, 3) <<
        432.4352, 0.0,      429.1616,
        0.0,      432.3150, 242.0728,
        0.0,      0.0,      1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.038960, 0.032735, 0.001603, 0.000783, 0.0)
    };

    calibrations_["/cam_2/color/image_raw"] = {
      (cv::Mat_<double>(3, 3) <<
        413.3101, 0.0,      424.8177,
        0.0,      413.8753, 248.7681,
        0.0,      0.0,      1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.045676, 0.033884, -0.001094, 0.000959, 0.0)
    };

    // ------------------------------------------------------------
    // Marker semantics
    // ------------------------------------------------------------
    marker_labels_[1] = "Medical";
    marker_labels_[2] = "Supply";

    // IMPORTANT:
    // This assumes your physical markers were generated from DICT_4X4_50.
    // If they came from a different 4x4 family, change this enum.
    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detector_params_ = cv::aruco::DetectorParameters();

    // ------------------------------------------------------------
    // Publishers
    // ------------------------------------------------------------
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/aruco/pose", 10);
    type_pub_ = this->create_publisher<std_msgs::msg::String>("/aruco/landing_place_type", 10);
    msg_pub_  = this->create_publisher<std_msgs::msg::String>("/aruco/message", 10);

    // ------------------------------------------------------------
    // Subscribers
    // ------------------------------------------------------------
    for (const auto & topic : camera_topics_) {
      auto sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic,
        10,
        [this, topic](const sensor_msgs::msg::Image::SharedPtr msg) {
          this->imageCallback(msg, topic);
        }
      );
      image_subs_.push_back(sub);
    }

    RCLCPP_INFO(this->get_logger(), "ArucoPoseEstimator C++ node started.");
  }

private:
  struct CameraCalibration
  {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
  };

  std::vector<std::string> camera_topics_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;

  std::map<std::string, cv::Matx44d> camera_to_drone_tf_;
  std::map<std::string, CameraCalibration> calibrations_;
  std::map<int, std::string> marker_labels_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr type_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr msg_pub_;

  cv::aruco::Dictionary aruco_dict_;
  cv::aruco::DetectorParameters detector_params_;

  double marker_length_;

  void imageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const std::string & topic_name)
  {
    cv::Mat frame;

    try {
      frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "CvBridge conversion failed for %s: %s",
        topic_name.c_str(), e.what());
      return;
    }

    const auto calib_it = calibrations_.find(topic_name);
    const auto tf_it = camera_to_drone_tf_.find(topic_name);

    if (calib_it == calibrations_.end() || tf_it == camera_to_drone_tf_.end()) {
      RCLCPP_ERROR(this->get_logger(), "Missing calibration or transform for topic: %s", topic_name.c_str());
      return;
    }

    const auto & camera_matrix = calib_it->second.camera_matrix;
    const auto & dist_coeffs = calib_it->second.dist_coeffs;
    const auto & T_drone_camera = tf_it->second;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::ArucoDetector detector(aruco_dict_, detector_params_);
    detector.detectMarkers(gray, corners, ids, rejected);

    if (ids.empty()) {
      return;
    }

    // Keep only marker IDs 1 and 2
    std::vector<std::vector<cv::Point2f>> valid_corners;
    std::vector<int> valid_ids;

    for (size_t i = 0; i < ids.size(); ++i) {
      if (marker_labels_.count(ids[i]) > 0) {
        valid_corners.push_back(corners[i]);
        valid_ids.push_back(ids[i]);
      }
    }

    if (valid_ids.empty()) {
      return;
    }

    // Pose of marker in camera frame: ^C T_M
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(
      valid_corners,
      marker_length_,
      camera_matrix,
      dist_coeffs,
      rvecs,
      tvecs
    );

    for (size_t i = 0; i < valid_ids.size(); ++i) {
      const int marker_id = valid_ids[i];
      const std::string landing_type = marker_labels_[marker_id];

      const cv::Matx44d T_camera_marker = rvecTvecToHomogeneous(rvecs[i], tvecs[i]);

      // ^D T_M = ^D T_C * ^C T_M
      // Published pose is therefore in the drone frame.
      const cv::Matx44d T_drone_marker = T_drone_camera * T_camera_marker;

      auto pose_msg = matrixToPoseStamped(T_drone_marker, msg->header.stamp);
      pose_pub_->publish(pose_msg);

      std_msgs::msg::String type_msg;
      type_msg.data = landing_type;
      type_pub_->publish(type_msg);

      std_msgs::msg::String info_msg;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3)
          << "Marker ID: " << marker_id
          << ", Type: " << landing_type
          << ", Source: " << topic_name
          << ", Frame: drone_base"
          << ", Position: ("
          << pose_msg.pose.position.x << ", "
          << pose_msg.pose.position.y << ", "
          << pose_msg.pose.position.z << ")";
      info_msg.data = oss.str();
      msg_pub_->publish(info_msg);
    }
  }

  cv::Matx44d rvecTvecToHomogeneous(const cv::Vec3d & rvec, const cv::Vec3d & tvec) const
  {
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    cv::Matx44d T = cv::Matx44d::eye();

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        T(r, c) = R_cv.at<double>(r, c);
      }
    }

    T(0, 3) = tvec[0];
    T(1, 3) = tvec[1];
    T(2, 3) = tvec[2];

    return T;
  }

  geometry_msgs::msg::PoseStamped matrixToPoseStamped(
    const cv::Matx44d & T,
    const builtin_interfaces::msg::Time & stamp) const
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "drone_base";

    pose_msg.pose.position.x = T(0, 3);
    pose_msg.pose.position.y = T(1, 3);
    pose_msg.pose.position.z = T(2, 3);

    cv::Matx33d R(
      T(0, 0), T(0, 1), T(0, 2),
      T(1, 0), T(1, 1), T(1, 2),
      T(2, 0), T(2, 1), T(2, 2)
    );

    auto q = rotationMatrixToQuaternion(R);

    pose_msg.pose.orientation.x = q[0];
    pose_msg.pose.orientation.y = q[1];
    pose_msg.pose.orientation.z = q[2];
    pose_msg.pose.orientation.w = q[3];

    return pose_msg;
  }

  std::array<double, 4> rotationMatrixToQuaternion(const cv::Matx33d & R) const
  {
    std::array<double, 4> q{};  // [x, y, z, w]
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);

    if (tr > 0.0) {
      double S = std::sqrt(tr + 1.0) * 2.0;
      q[3] = 0.25 * S;
      q[0] = (R(2, 1) - R(1, 2)) / S;
      q[1] = (R(0, 2) - R(2, 0)) / S;
      q[2] = (R(1, 0) - R(0, 1)) / S;
    } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
      double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
      q[3] = (R(2, 1) - R(1, 2)) / S;
      q[0] = 0.25 * S;
      q[1] = (R(0, 1) + R(1, 0)) / S;
      q[2] = (R(0, 2) + R(2, 0)) / S;
    } else if (R(1, 1) > R(2, 2)) {
      double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
      q[3] = (R(0, 2) - R(2, 0)) / S;
      q[0] = (R(0, 1) + R(1, 0)) / S;
      q[1] = 0.25 * S;
      q[2] = (R(1, 2) + R(2, 1)) / S;
    } else {
      double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
      q[3] = (R(1, 0) - R(0, 1)) / S;
      q[0] = (R(0, 2) + R(2, 0)) / S;
      q[1] = (R(1, 2) + R(2, 1)) / S;
      q[2] = 0.25 * S;
    }

    const double norm =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

    if (norm > 1e-12) {
      q[0] /= norm;
      q[1] /= norm;
      q[2] /= norm;
      q[3] /= norm;
    }

    return q;
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoPoseEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
