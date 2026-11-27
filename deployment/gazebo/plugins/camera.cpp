#include <cmath>
#include <memory>
#include <gazebo/gazebo.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

namespace gazebo {
class SigmaRLCamera : public SensorPlugin {
 public:
  ~SigmaRLCamera() override { frame_.reset(); }

  void Load(sensors::SensorPtr sensor, sdf::ElementPtr) override {
    if (!ros::isInitialized()) {
      gzerr << "SigmaRLCamera requires libgazebo_ros_api_plugin.so\n";
      return;
    }
    sensor_ = std::dynamic_pointer_cast<sensors::CameraSensor>(sensor);
    if (!sensor_ || !sensor_->Camera()) {
      gzerr << "SigmaRLCamera must be attached to a camera sensor\n";
      return;
    }
    node_.reset(new ros::NodeHandle("/overhead_camera"));
    images_ = node_->advertise<sensor_msgs::Image>("image_raw", 1);
    info_ = node_->advertise<sensor_msgs::CameraInfo>("camera_info", 1, true);
    frame_ = sensor_->Camera()->ConnectNewImageFrame(
        std::bind(&SigmaRLCamera::Frame, this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
    sensor_->SetActive(true);
    gzmsg << "SigmaRLCamera ready: /overhead_camera/image_raw\n";
  }

 private:
  void Frame(const unsigned char *data, unsigned width, unsigned height,
             unsigned depth, const std::string &format) {
    if (depth != 3 || format != "R8G8B8") return;
    sensor_msgs::CameraInfo info;
    info.header.stamp.fromSec(sensor_->LastMeasurementTime().Double());
    info.header.frame_id = "overhead_camera_optical";
    info.width = width;
    info.height = height;
    const double focal = width / (2 * std::tan(sensor_->Camera()->HFOV().Radian() / 2));
    const double cx = (width - 1) / 2.0, cy = (height - 1) / 2.0;
    info.distortion_model = "plumb_bob";
    info.D.assign(5, 0.0);
    info.K = {{focal, 0, cx, 0, focal, cy, 0, 0, 1}};
    info.R = {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    info.P = {{focal, 0, cx, 0, 0, focal, cy, 0, 0, 0, 1, 0}};
    info_.publish(info);
    if (images_.getNumSubscribers() == 0) return;
    sensor_msgs::Image image;
    image.header = info.header;
    image.height = height;
    image.width = width;
    image.encoding = "rgb8";
    image.is_bigendian = false;
    image.step = width * depth;
    image.data.assign(data, data + height * image.step);
    images_.publish(image);
  }

  sensors::CameraSensorPtr sensor_;
  event::ConnectionPtr frame_;
  std::unique_ptr<ros::NodeHandle> node_;
  ros::Publisher images_, info_;
};
GZ_REGISTER_SENSOR_PLUGIN(SigmaRLCamera)
}  // namespace gazebo
