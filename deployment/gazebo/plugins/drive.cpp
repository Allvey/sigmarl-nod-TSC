#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

namespace gazebo {
class SigmaRLDrive : public ModelPlugin {
 public:
  ~SigmaRLDrive() override {
    update_.reset();
    if (spinner_) spinner_->stop();
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    if (!ros::isInitialized()) {
      gzerr << "SigmaRLDrive requires libgazebo_ros_api_plugin.so\n";
      return;
    }
    model_ = model;
    ns_ = sdf->Get<std::string>("namespace");
    wheelbase_ = sdf->Get<double>("wheelbase");
    track_ = sdf->Get<double>("track");
    radius_ = sdf->Get<double>("wheel_radius");
    max_speed_ = sdf->Get<double>("max_speed");
    max_steer_ = sdf->Get<double>("max_steering");
    max_accel_ = sdf->Get<double>("max_acceleration");
    max_steer_rate_ = sdf->Get<double>("max_steering_rate");
    timeout_ = sdf->Get<double>("command_timeout");
    rear_left_ = model->GetJoint("rear_left_axle");
    rear_right_ = model->GetJoint("rear_right_axle");
    front_left_ = model->GetJoint("front_left_steer");
    front_right_ = model->GetJoint("front_right_steer");
    if (!rear_left_ || !rear_right_ || !front_left_ || !front_right_) {
      gzerr << "SigmaRLDrive: missing vehicle joints\n";
      return;
    }
    node_.reset(new ros::NodeHandle("/" + ns_));
    node_->setCallbackQueue(&queue_);
    command_ = node_->subscribe("drive", 1, &SigmaRLDrive::Command, this);
    truth_ = node_->advertise<nav_msgs::Odometry>("ground_truth", 1);
    spinner_.reset(new ros::AsyncSpinner(1, &queue_));
    spinner_->start();
    previous_ = model->GetWorld()->SimTime().Double();
    update_ = event::Events::ConnectWorldUpdateBegin(std::bind(&SigmaRLDrive::Update, this));
    gzmsg << "SigmaRLDrive ready: /" << ns_ << "/drive (x=m/s, y=steering radians)\n";
  }

  void Reset() override {
    std::lock_guard<std::mutex> guard(mutex_);
    target_speed_ = target_steer_ = speed_ = steer_ = 0;
    have_command_ = false;
    previous_ = last_publish_ = 0;
  }

 private:
  static double Limit(double value, double maximum) {
    return std::max(-maximum, std::min(value, maximum));
  }

  void Command(const geometry_msgs::Vector3Stamped::ConstPtr &message) {
    if (!std::isfinite(message->vector.x) || !std::isfinite(message->vector.y)) return;
    std::lock_guard<std::mutex> guard(mutex_);
    target_speed_ = Limit(message->vector.x, max_speed_);
    target_steer_ = Limit(message->vector.y, max_steer_);
    received_ = ros::WallTime::now();
    have_command_ = true;
  }

  void Update() {
    const double now = model_->GetWorld()->SimTime().Double();
    const double dt = now - previous_;
    previous_ = now;
    if (dt <= 0 || dt > 0.1) return;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!have_command_ || (ros::WallTime::now() - received_).toSec() > timeout_) {
        target_speed_ = 0;
        target_steer_ = 0;
      }
      speed_ += Limit(target_speed_ - speed_, max_accel_ * dt);
      steer_ += Limit(target_steer_ - steer_, max_steer_rate_ * dt);
    }
    const double curvature = std::tan(steer_) / wheelbase_;
    const double beta = std::atan(0.5 * std::tan(steer_));
    const double rear_speed = speed_ * std::cos(beta);
    Motor(rear_left_, rear_speed * (1 - track_ * curvature / 2) / radius_, 0.05);
    Motor(rear_right_, rear_speed * (1 + track_ * curvature / 2) / radius_, 0.05);
    Servo(front_left_, std::atan2(wheelbase_ * curvature, 1 - track_ * curvature / 2));
    Servo(front_right_, std::atan2(wheelbase_ * curvature, 1 + track_ * curvature / 2));
    if (now < last_publish_) last_publish_ = 0;
    if (now - last_publish_ >= 0.05) {
      PublishTruth(now);
      last_publish_ = now;
    }
  }

  static void Motor(const physics::JointPtr &joint, double velocity, double force) {
    joint->SetParam("fmax", 0, force);
    joint->SetParam("vel", 0, velocity);
  }

  static void Servo(const physics::JointPtr &joint, double angle) {
    Motor(joint, Limit(12 * (angle - joint->Position(0)), 3.0), 0.1);
  }

  void PublishTruth(double now) {
    const auto pose = model_->WorldPose();
    const auto velocity = pose.Rot().RotateVectorReverse(model_->WorldLinearVel());
    const auto angular = pose.Rot().RotateVectorReverse(model_->WorldAngularVel());
    nav_msgs::Odometry message;
    message.header.stamp.fromSec(now);
    message.header.frame_id = "map";
    message.child_frame_id = ns_ + "/base_link";
    message.pose.pose.position.x = pose.Pos().X();
    message.pose.pose.position.y = pose.Pos().Y();
    message.pose.pose.position.z = pose.Pos().Z();
    message.pose.pose.orientation.x = pose.Rot().X();
    message.pose.pose.orientation.y = pose.Rot().Y();
    message.pose.pose.orientation.z = pose.Rot().Z();
    message.pose.pose.orientation.w = pose.Rot().W();
    message.twist.twist.linear.x = velocity.X();
    message.twist.twist.linear.y = velocity.Y();
    message.twist.twist.linear.z = velocity.Z();
    message.twist.twist.angular.x = angular.X();
    message.twist.twist.angular.y = angular.Y();
    message.twist.twist.angular.z = angular.Z();
    truth_.publish(message);
  }

  physics::ModelPtr model_;
  physics::JointPtr rear_left_, rear_right_, front_left_, front_right_;
  event::ConnectionPtr update_;
  ros::CallbackQueue queue_;
  std::unique_ptr<ros::NodeHandle> node_;
  std::unique_ptr<ros::AsyncSpinner> spinner_;
  ros::Subscriber command_;
  ros::Publisher truth_;
  std::mutex mutex_;
  std::string ns_;
  ros::WallTime received_;
  bool have_command_ = false;
  double target_speed_ = 0, target_steer_ = 0, speed_ = 0, steer_ = 0;
  double previous_ = 0, last_publish_ = 0;
  double wheelbase_, track_, radius_, max_speed_, max_steer_, max_accel_, max_steer_rate_, timeout_;
};
GZ_REGISTER_MODEL_PLUGIN(SigmaRLDrive)
}  // namespace gazebo
