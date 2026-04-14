#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class ArucoPoseEstimator : public rclcpp::Node
{
public:
  ArucoPoseEstimator()
  : Node("aruco_pose_estimator")
  {
    marker_length_ = this->declare_parameter<double>("marker_length", 1.0);
    drone_frame_id_ = this->declare_parameter<std::string>("drone_frame_id", "drone_base");
    processing_period_ms_ = this->declare_parameter<int>("processing_period_ms", 33);

    marker_labels_[1] = "Medical";
    marker_labels_[2] = "Supply";

    // Your OpenCV ArUco API is mixed:
    // - getPredefinedDictionary returns a value type
    // - detectMarkers expects cv::Ptr<Dictionary> and cv::Ptr<DetectorParameters>
    // So wrap value objects with cv::makePtr.
    aruco_dict_ = cv::makePtr<cv::aruco::Dictionary>(
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
    detector_params_ = cv::makePtr<cv::aruco::DetectorParameters>(
      cv::aruco::DetectorParameters());

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/aruco/pose", rclcpp::QoS(10));
    type_pub_ = this->create_publisher<std_msgs::msg::String>("/aruco/landing_place_type", rclcpp::QoS(10));
    msg_pub_  = this->create_publisher<std_msgs::msg::String>("/aruco/message", rclcpp::QoS(10));

    addCamera(
      "/cam_1/color/image_raw",
      cv::Matx44d(
         0.0,  0.0,  1.0,   0.0,
        -1.0,  0.0,  0.0,  -0.12037,
         0.0, -1.0,  0.0,  -0.11435,
         0.0,  0.0,  0.0,   1.0
      ),
      (cv::Mat_<double>(3, 3) <<
        432.4352, 0.0,      429.1616,
        0.0,      432.3150, 242.0728,
        0.0,      0.0,      1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.038960, 0.032735, 0.001603, 0.000783, 0.0)
    );

    addCamera(
      "/cam_2/color/image_raw",
      cv::Matx44d(
         0.0,  0.70710678118,  0.70710678118,   0.0,
         1.0,  0.0,            0.0,            -0.12037,
         0.0,  0.70710678118, -0.70710678118, -0.11435,
         0.0,  0.0,            0.0,             1.0
      ),
      (cv::Mat_<double>(3, 3) <<
        413.3101, 0.0,      424.8177,
        0.0,      413.8753, 248.7681,
        0.0,      0.0,      1.0),
      (cv::Mat_<double>(1, 5) <<
        -0.045676, 0.033884, -0.001094, 0.000959, 0.0)
    );

    const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);

    for (auto & [topic, camera] : cameras_) {
      camera->sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic,
        image_qos,
        [this, topic](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          this->storeLatestImage(msg, topic);
        }
      );
    }

    processing_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(processing_period_ms_),
      [this]() { this->processLatestFrames(); });

    RCLCPP_INFO(
      this->get_logger(),
      "ArucoPoseEstimator started. marker_length=%.6f, frame_id=%s, period=%d ms",
      marker_length_, drone_frame_id_.c_str(), processing_period_ms_);
  }

private:
  struct CameraCalibration
  {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
  };

  struct CameraState
  {
    CameraCalibration calibration;
    cv::Matx44d T_drone_camera = cv::Matx44d::eye();

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub;

    sensor_msgs::msg::Image::ConstSharedPtr latest_msg;
    std::mutex latest_msg_mutex;
  };

  std::map<std::string, std::shared_ptr<CameraState>> cameras_;
  std::map<int, std::string> marker_labels_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr type_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr msg_pub_;
  rclcpp::TimerBase::SharedPtr processing_timer_;

  cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
  cv::Ptr<cv::aruco::DetectorParameters> detector_params_;

  double marker_length_{1.0};
  std::string drone_frame_id_{"drone_base"};
  int processing_period_ms_{33};

  void addCamera(
    const std::string & topic,
    const cv::Matx44d & T_drone_camera,
    const cv::Mat & camera_matrix,
    const cv::Mat & dist_coeffs)
  {
    auto camera = std::make_shared<CameraState>();
    camera->T_drone_camera = T_drone_camera;
    camera->calibration.camera_matrix = camera_matrix.clone();
    camera->calibration.dist_coeffs = dist_coeffs.clone();
    cameras_[topic] = camera;
  }

  void storeLatestImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & topic_name)
  {
    const auto it = cameras_.find(topic_name);
    if (it == cameras_.end()) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Received image for unknown topic: %s", topic_name.c_str());
      return;
    }

    std::lock_guard<std::mutex> lock(it->second->latest_msg_mutex);
    it->second->latest_msg = msg;
  }

  void processLatestFrames()
  {
    for (auto & [topic_name, camera] : cameras_) {
      sensor_msgs::msg::Image::ConstSharedPtr msg;

      {
        std::lock_guard<std::mutex> lock(camera->latest_msg_mutex);
        if (!camera->latest_msg) {
          continue;
        }
        msg = camera->latest_msg;
        camera->latest_msg.reset();
      }

      processImage(msg, topic_name, *camera);
    }
  }

  void processImage(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & topic_name,
    const CameraState & camera)
  {
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "cv_bridge failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    cv::Mat gray;

    try {
      const auto & enc = msg->encoding;

      if (enc == sensor_msgs::image_encodings::MONO8) {
        gray = cv_ptr->image;
      } else if (enc == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
      } else if (enc == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGB2GRAY);
      } else if (enc == sensor_msgs::image_encodings::BGRA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGRA2GRAY);
      } else if (enc == sensor_msgs::image_encodings::RGBA8) {
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGBA2GRAY);
      } else {
        auto bgr_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(bgr_ptr->image, gray, cv::COLOR_BGR2GRAY);
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Image conversion failed for %s: %s", topic_name.c_str(), e.what());
      return;
    }

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, detector_params_, rejected);

    if (ids.empty()) {
      return;
    }

    std::vector<std::vector<cv::Point2f>> valid_corners;
    std::vector<int> valid_ids;
    valid_corners.reserve(ids.size());
    valid_ids.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
      if (marker_labels_.find(ids[i]) != marker_labels_.end()) {
        valid_corners.push_back(corners[i]);
        valid_ids.push_back(ids[i]);
      }
    }

    if (valid_ids.empty()) {
      return;
    }

    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;

    cv::aruco::estimatePoseSingleMarkers(
      valid_corners,
      marker_length_,
      camera.calibration.camera_matrix,
      camera.calibration.dist_coeffs,
      rvecs,
      tvecs
    );

    for (size_t i = 0; i < valid_ids.size(); ++i) {
      const int marker_id = valid_ids[i];
      const auto label_it = marker_labels_.find(marker_id);
      if (label_it == marker_labels_.end()) {
        continue;
      }

      const std::string & landing_type = label_it->second;

      const cv::Matx44d T_camera_marker = rvecTvecToHomogeneous(rvecs[i], tvecs[i]);
      const cv::Matx44d T_drone_marker = camera.T_drone_camera * T_camera_marker;

      auto pose_msg = matrixToPoseStamped(T_drone_marker, msg->header.stamp);
      pose_pub_->publish(pose_msg);

      if (type_pub_->get_subscription_count() > 0U) {
        std_msgs::msg::String type_msg;
        type_msg.data = landing_type;
        type_pub_->publish(type_msg);
      }

      if (msg_pub_->get_subscription_count() > 0U) {
        std_msgs::msg::String info_msg;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "Marker ID: " << marker_id
            << ", Type: " << landing_type
            << ", Source: " << topic_name
            << ", Frame: " << drone_frame_id_
            << ", Position: ("
            << pose_msg.pose.position.x << ", "
            << pose_msg.pose.position.y << ", "
            << pose_msg.pose.position.z << ")";
        info_msg.data = oss.str();
        msg_pub_->publish(info_msg);
      }
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
    pose_msg.header.frame_id = drone_frame_id_;

    pose_msg.pose.position.x = T(0, 3);
    pose_msg.pose.position.y = T(1, 3);
    pose_msg.pose.position.z = T(2, 3);

    cv::Matx33d R(
      T(0, 0), T(0, 1), T(0, 2),
      T(1, 0), T(1, 1), T(1, 2),
      T(2, 0), T(2, 1), T(2, 2)
    );

    const auto q = rotationMatrixToQuaternion(R);

    pose_msg.pose.orientation.x = q[0];
    pose_msg.pose.orientation.y = q[1];
    pose_msg.pose.orientation.z = q[2];
    pose_msg.pose.orientation.w = q[3];

    return pose_msg;
  }

  std::array<double, 4> rotationMatrixToQuaternion(const cv::Matx33d & R) const
  {
    std::array<double, 4> q{};
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);

    if (tr > 0.0) {
      const double S = std::sqrt(tr + 1.0) * 2.0;
      q[3] = 0.25 * S;
      q[0] = (R(2, 1) - R(1, 2)) / S;
      q[1] = (R(0, 2) - R(2, 0)) / S;
      q[2] = (R(1, 0) - R(0, 1)) / S;
    } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
      const double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
      q[3] = (R(2, 1) - R(1, 2)) / S;
      q[0] = 0.25 * S;
      q[1] = (R(0, 1) + R(1, 0)) / S;
      q[2] = (R(0, 2) + R(2, 0)) / S;
    } else if (R(1, 1) > R(2, 2)) {
      const double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
      q[3] = (R(0, 2) - R(2, 0)) / S;
      q[0] = (R(0, 1) + R(1, 0)) / S;
      q[1] = 0.25 * S;
      q[2] = (R(1, 2) + R(2, 1)) / S;
    } else {
      const double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
      q[3] = (R(1, 0) - R(0, 1)) / S;
      q[0] = (R(0, 2) + R(2, 0)) / S;
      q[1] = (R(1, 2) + R(2, 1)) / S;
      q[2] = 0.25 * S;
    }

    const double norm = std::sqrt(
      q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

    if (norm > 1e-12) {
      q[0] /= norm;
      q[1] /= norm;
      q[2] /= norm;
      q[3] /= norm;
    } else {
      q = {0.0, 0.0, 0.0, 1.0};
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
