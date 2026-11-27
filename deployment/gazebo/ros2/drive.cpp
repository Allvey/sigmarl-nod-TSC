#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace gazebo {
// Manufacturer-style planar kinematics, with bounded commands and a wall-clock watchdog.
// This does not model individual Mecanum rollers or tyre/ground slip.
class JetAutoROS2 : public ModelPlugin {
 public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    node_ = gazebo_ros::Node::Get(sdf);
    const auto command_topic = sdf->Get<std::string>("command_topic");
    base_frame_ = sdf->Get<std::string>("base_frame");
    lidar_frame_ = sdf->Get<std::string>("lidar_frame");
    lidar_xyz_ = sdf->Get<ignition::math::Vector3d>("lidar_xyz");
    lidar_yaw_ = sdf->Get<double>("lidar_yaw");
    max_x_ = sdf->Get<double>("max_speed");
    max_y_ = sdf->Get<double>("max_lateral_speed");
    max_w_ = sdf->Get<double>("max_yaw_rate");
    acceleration_ = sdf->Get<double>("max_acceleration");
    angular_acceleration_ = sdf->Get<double>("max_yaw_acceleration");
    timeout_ = sdf->Get<double>("command_timeout");
    subscription_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        command_topic, rclcpp::QoS(1),
        [this](geometry_msgs::msg::Twist::SharedPtr command) {
          if (!std::isfinite(command->linear.x) || !std::isfinite(command->linear.y) ||
              !std::isfinite(command->angular.z)) return;
          std::lock_guard<std::mutex> lock(mutex_);
          target_x_ = Limit(command->linear.x, max_x_);
          target_y_ = Limit(command->linear.y, max_y_);
          target_w_ = Limit(command->angular.z, max_w_);
          received_ = Clock::now();
          have_command_ = true;
        });
    odometry_ = node_->create_publisher<nav_msgs::msg::Odometry>("ground_truth", rclcpp::QoS(1));
    transforms_ = node_->create_publisher<tf2_msgs::msg::TFMessage>("/tf", rclcpp::QoS(100));
    last_time_ = model_->GetWorld()->SimTime().Double();
    connection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&JetAutoROS2::Update, this));
    RCLCPP_INFO(node_->get_logger(), "JetAuto planar controller ready: %s/%s", node_->get_namespace(), command_topic.c_str());
  }

  void Reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_x_ = target_y_ = target_w_ = x_ = y_ = w_ = 0;
    have_command_ = false;
    last_time_ = last_publish_ = 0;
  }

 private:
  using Clock = std::chrono::steady_clock;
  static double Limit(double value, double maximum) { return std::max(-maximum, std::min(value, maximum)); }

  void Update() {
    const double now = model_->GetWorld()->SimTime().Double();
    const double dt = now - last_time_;
    last_time_ = now;
    if (dt <= 0 || dt > 0.1) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!have_command_ || std::chrono::duration<double>(Clock::now() - received_).count() > timeout_)
        target_x_ = target_y_ = target_w_ = 0;
      x_ += Limit(target_x_ - x_, acceleration_ * dt);
      y_ += Limit(target_y_ - y_, acceleration_ * dt);
      w_ += Limit(target_w_ - w_, angular_acceleration_ * dt);
    }
    const auto pose = model_->WorldPose();
    const double yaw = pose.Rot().Yaw();
    const ignition::math::Vector3d translation(std::cos(yaw) * x_ - std::sin(yaw) * y_,
                                              std::sin(yaw) * x_ + std::cos(yaw) * y_, 0);
    const ignition::math::Vector3d omega(0, 0, w_);
    // Twist is defined at base_footprint, while Gazebo link velocities are at
    // each centre of mass. Apply v_CoG = v_base + omega x r consistently.
    for (const auto &link : model_->GetLinks()) {
      const auto offset = link->WorldCoGPose().Pos() - pose.Pos();
      auto velocity = translation + omega.Cross(offset);
      velocity.Z() = link->WorldCoGLinearVel().Z();  // Preserve gravity/contact response.
      const auto angular = link->WorldAngularVel();
      link->SetLinearVel(velocity);
      link->SetAngularVel({angular.X(), angular.Y(), w_});
    }
    if (now < last_publish_) last_publish_ = 0;
    if (now - last_publish_ < 0.05) return;
    last_publish_ = now;
    nav_msgs::msg::Odometry message;
    message.header.stamp = rclcpp::Time(static_cast<int64_t>(now * 1e9));
    message.header.frame_id = "map";
    message.child_frame_id = base_frame_;
    message.pose.pose.position.x = pose.Pos().X();
    message.pose.pose.position.y = pose.Pos().Y();
    message.pose.pose.position.z = pose.Pos().Z();
    message.pose.pose.orientation.x = pose.Rot().X();
    message.pose.pose.orientation.y = pose.Rot().Y();
    message.pose.pose.orientation.z = pose.Rot().Z();
    message.pose.pose.orientation.w = pose.Rot().W();
    const auto canonical = model_->GetLink();
    const auto offset = canonical->WorldCoGPose().Pos() - pose.Pos();
    const auto root_velocity = canonical->WorldCoGLinearVel() - canonical->WorldAngularVel().Cross(offset);
    const auto body_velocity = pose.Rot().RotateVectorReverse(root_velocity);
    message.twist.twist.linear.x = body_velocity.X();
    message.twist.twist.linear.y = body_velocity.Y();
    message.twist.twist.angular.z = model_->WorldAngularVel().Z();
    odometry_->publish(message);
    tf2_msgs::msg::TFMessage frames;
    geometry_msgs::msg::TransformStamped base;
    base.header = message.header;
    base.child_frame_id = base_frame_;
    base.transform.translation.x = pose.Pos().X();
    base.transform.translation.y = pose.Pos().Y();
    base.transform.translation.z = pose.Pos().Z();
    base.transform.rotation = message.pose.pose.orientation;
    frames.transforms.push_back(base);
    geometry_msgs::msg::TransformStamped laser;
    laser.header.stamp = message.header.stamp;
    laser.header.frame_id = base_frame_;
    laser.child_frame_id = lidar_frame_;
    laser.transform.translation.x = lidar_xyz_.X();
    laser.transform.translation.y = lidar_xyz_.Y();
    laser.transform.translation.z = lidar_xyz_.Z();
    laser.transform.rotation.z = std::sin(lidar_yaw_ / 2);
    laser.transform.rotation.w = std::cos(lidar_yaw_ / 2);
    frames.transforms.push_back(laser);
    transforms_->publish(frames);
  }

  physics::ModelPtr model_;
  gazebo_ros::Node::SharedPtr node_;
  event::ConnectionPtr connection_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr transforms_;
  std::mutex mutex_;
  std::string base_frame_, lidar_frame_;
  ignition::math::Vector3d lidar_xyz_;
  double lidar_yaw_ = 0;
  Clock::time_point received_;
  bool have_command_ = false;
  double x_ = 0, y_ = 0, w_ = 0, target_x_ = 0, target_y_ = 0, target_w_ = 0;
  double last_time_ = 0, last_publish_ = 0;
  double max_x_, max_y_, max_w_, acceleration_, angular_acceleration_, timeout_;
};
GZ_REGISTER_MODEL_PLUGIN(JetAutoROS2)
}  // namespace gazebo
