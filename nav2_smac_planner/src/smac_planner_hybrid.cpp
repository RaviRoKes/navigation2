// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include "tf2/utils.h"

#include "Eigen/Core"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "nav2_smac_planner/direction_map.hpp" 
#include "visualization_msgs/msg/marker_array.hpp"


// #define BENCHMARK_TESTING

namespace nav2_smac_planner
{

using namespace std::chrono;  // NOLINT
using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;

SmacPlannerHybrid::SmacPlannerHybrid()
: _a_star(nullptr),
  _collision_checker(nullptr, 1, nullptr),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_downsampler(nullptr)
{
}

SmacPlannerHybrid::~SmacPlannerHybrid()
{
  RCLCPP_INFO(
    _logger, "Destroying plugin %s of type SmacPlannerHybrid",
    _name.c_str());
}

void SmacPlannerHybrid::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer>/*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  _node = parent;
  auto node = parent.lock();
  _logger = node->get_logger();
  _clock = node->get_clock();
  _costmap = costmap_ros->getCostmap();
  _costmap_ros = costmap_ros;
  _name = name;
  _global_frame = costmap_ros->getGlobalFrameID();

  RCLCPP_INFO(_logger, "Configuring %s of type SmacPlannerHybrid", name.c_str());

  int angle_quantizations;
  double analytic_expansion_max_length_m;
  bool smooth_path;

  // General planner params
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".downsample_costmap", rclcpp::ParameterValue(false));
  node->get_parameter(name + ".downsample_costmap", _downsample_costmap);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".downsampling_factor", rclcpp::ParameterValue(1));
  node->get_parameter(name + ".downsampling_factor", _downsampling_factor);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".angle_quantization_bins", rclcpp::ParameterValue(72));
  node->get_parameter(name + ".angle_quantization_bins", angle_quantizations);
  _angle_bin_size = 2.0 * M_PI / angle_quantizations;
  _angle_quantizations = static_cast<unsigned int>(angle_quantizations);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".tolerance", rclcpp::ParameterValue(0.25));
  _tolerance = static_cast<float>(node->get_parameter(name + ".tolerance").as_double());
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".allow_unknown", rclcpp::ParameterValue(true));
  node->get_parameter(name + ".allow_unknown", _allow_unknown);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_iterations", rclcpp::ParameterValue(1000000));
  node->get_parameter(name + ".max_iterations", _max_iterations);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_on_approach_iterations", rclcpp::ParameterValue(1000));
  node->get_parameter(name + ".max_on_approach_iterations", _max_on_approach_iterations);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".smooth_path", rclcpp::ParameterValue(true));
  node->get_parameter(name + ".smooth_path", smooth_path);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".minimum_turning_radius", rclcpp::ParameterValue(0.4));
  node->get_parameter(name + ".minimum_turning_radius", _minimum_turning_radius_global_coords);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".cache_obstacle_heuristic", rclcpp::ParameterValue(false));
  node->get_parameter(name + ".cache_obstacle_heuristic", _search_info.cache_obstacle_heuristic);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".reverse_penalty", rclcpp::ParameterValue(2.0));
  node->get_parameter(name + ".reverse_penalty", _search_info.reverse_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".change_penalty", rclcpp::ParameterValue(0.0));
  node->get_parameter(name + ".change_penalty", _search_info.change_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".non_straight_penalty", rclcpp::ParameterValue(1.2));
  node->get_parameter(name + ".non_straight_penalty", _search_info.non_straight_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".cost_penalty", rclcpp::ParameterValue(2.0));
  node->get_parameter(name + ".cost_penalty", _search_info.cost_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".retrospective_penalty", rclcpp::ParameterValue(0.015));
  node->get_parameter(name + ".retrospective_penalty", _search_info.retrospective_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".analytic_expansion_ratio", rclcpp::ParameterValue(3.5));
  node->get_parameter(name + ".analytic_expansion_ratio", _search_info.analytic_expansion_ratio);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".analytic_expansion_max_length", rclcpp::ParameterValue(3.0));
  node->get_parameter(name + ".analytic_expansion_max_length", analytic_expansion_max_length_m);
  _search_info.analytic_expansion_max_length =
    analytic_expansion_max_length_m / _costmap->getResolution();

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_planning_time", rclcpp::ParameterValue(5.0));
  node->get_parameter(name + ".max_planning_time", _max_planning_time);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".lookup_table_size", rclcpp::ParameterValue(20.0));
  node->get_parameter(name + ".lookup_table_size", _lookup_table_size);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".motion_model_for_search", rclcpp::ParameterValue(std::string("DUBIN")));
  node->get_parameter(name + ".motion_model_for_search", _motion_model_for_search);
  _motion_model = fromString(_motion_model_for_search);
  if (_motion_model == MotionModel::UNKNOWN) {
    RCLCPP_WARN(
      _logger,
      "Unable to get MotionModel search type. Given '%s', "
      "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP, STATE_LATTICE.",
      _motion_model_for_search.c_str());
  }

  if (_max_on_approach_iterations <= 0) {
    RCLCPP_INFO(
      _logger, "On approach iteration selected as <= 0, "
      "disabling tolerance and on approach iterations.");
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    RCLCPP_INFO(
      _logger, "maximum iteration selected as <= 0, "
      "disabling maximum iterations.");
    _max_iterations = std::numeric_limits<int>::max();
  }

  // convert to grid coordinates
  if (!_downsample_costmap) {
    _downsampling_factor = 1;
  }
  _search_info.minimum_turning_radius =
    _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
  _lookup_table_dim =
    static_cast<float>(_lookup_table_size) /
    static_cast<float>(_costmap->getResolution() * _downsampling_factor);

  // Make sure its a whole number
  _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

  // Make sure its an odd number
  if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
    RCLCPP_INFO(
      _logger,
      "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
      _lookup_table_dim);
    _lookup_table_dim += 1.0;
  }

  // --- Direction-map guidance parameters ---
  nav2_util::declare_parameter_if_not_declared(
      node, name + ".use_direction_map", rclcpp::ParameterValue(false));
  node->get_parameter(name + ".use_direction_map", use_direction_map_);

  nav2_util::declare_parameter_if_not_declared(
      node, name + ".direction_map_topic", rclcpp::ParameterValue(std::string("direction_map")));
  node->get_parameter(name + ".direction_map_topic", direction_map_topic_);

  nav2_util::declare_parameter_if_not_declared(
      node, name + ".direction_heading_weight", rclcpp::ParameterValue(0.0));
  node->get_parameter(name + ".direction_heading_weight", direction_heading_weight_);

  // Create and subscribe to the direction map if enabled
  if (use_direction_map_)
  {
    if (!direction_map_)
    {
      direction_map_ = std::make_shared<nav2_smac_planner::DirectionMap>();
    }
    _search_info.direction_map = direction_map_; // keep SearchInfo in sync

    direction_map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
        direction_map_topic_,
        rclcpp::QoS(1).transient_local().reliable(),
        std::bind(&SmacPlannerHybrid::directionMapCallback, this, std::placeholders::_1));

    // THIS MARKER PUBLISHER H
    heading_marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "direction_map_vectors", rclcpp::QoS(1).transient_local());
  }
  else
  {
    direction_map_sub_.reset();
    _search_info.direction_map.reset();
  }

      // Mirror all of these into SearchInfo (what NodeHybrid reads)
  _search_info.use_direction_map = use_direction_map_;
  _search_info.direction_map = direction_map_; // may be nullptr now; set below if used
  _search_info.direction_heading_weight = direction_heading_weight_;
  // Make the SearchInfo visible: SearchInfo passed to NodeHybrid
  NodeHybrid::setSearchInfo(&_search_info);


  // Initialize collision checker
  _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations, node);
  _collision_checker.setFootprint(
    _costmap_ros->getRobotFootprint(),
    _costmap_ros->getUseRadius(),
    findCircumscribedCost(_costmap_ros));

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _max_planning_time,
    _lookup_table_dim,
    _angle_quantizations);

  // Initialize path smoother
  if (smooth_path) {
    SmootherParams params;
    params.get(node, name);
    _smoother = std::make_unique<Smoother>(params);
    _smoother->initialize(_minimum_turning_radius_global_coords);
  }

  // Initialize costmap downsampler
  if (_downsample_costmap && _downsampling_factor > 1) {
    _costmap_downsampler = std::make_unique<CostmapDownsampler>();
    std::string topic_name = "downsampled_costmap";
    _costmap_downsampler->on_configure(
      node, _global_frame, topic_name, _costmap, _downsampling_factor);
  }

  _raw_plan_publisher = node->create_publisher<nav_msgs::msg::Path>("unsmoothed_plan", 1);

  RCLCPP_INFO(
    _logger, "Configured plugin %s of type SmacPlannerHybrid with "
    "maximum iterations %i, max on approach iterations %i, and %s. Tolerance %.2f."
    "Using motion model: %s.",
    _name.c_str(), _max_iterations, _max_on_approach_iterations,
    _allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal",
    _tolerance, toString(_motion_model).c_str());
}

void SmacPlannerHybrid::activate()
{
  RCLCPP_INFO(
    _logger, "Activating plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _raw_plan_publisher->on_activate();
  if (heading_marker_pub_)
  {
    heading_marker_pub_->on_activate();
  }
  if (_costmap_downsampler) {
    _costmap_downsampler->on_activate();
  }
  auto node = _node.lock();
  // Add callback for dynamic parameters
  _dyn_params_handler = node->add_on_set_parameters_callback(
    std::bind(&SmacPlannerHybrid::dynamicParametersCallback, this, _1));
}

void SmacPlannerHybrid::deactivate()
{
  RCLCPP_INFO(
    _logger, "Deactivating plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _raw_plan_publisher->on_deactivate();
  if (heading_marker_pub_)
  {
    heading_marker_pub_->on_deactivate();
  }

  if (_costmap_downsampler) {
    _costmap_downsampler->on_deactivate();
  }
  _dyn_params_handler.reset();
}

void SmacPlannerHybrid::cleanup()
{
  RCLCPP_INFO(
    _logger, "Cleaning up plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _a_star.reset();
  _smoother.reset();
  if (_costmap_downsampler) {
    _costmap_downsampler->on_cleanup();
    _costmap_downsampler.reset();
  }
  _raw_plan_publisher.reset();
  if (heading_marker_pub_)
  {
    heading_marker_pub_.reset();
  }
}

void SmacPlannerHybrid::directionMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  RCLCPP_INFO(_logger, "[DirectionMap] Directionmap callback ....Received map with size: %u x %u",
              msg->info.width, msg->info.height);

  if (!direction_map_)
  {
    direction_map_ = std::make_shared<DirectionMap>();
  }
  direction_map_->setGrid(*msg);
  {
    const auto &info = msg->info;
    const size_t width = info.width;
    const size_t height = info.height;
    const double resolution = info.resolution;
    const double origin_x = info.origin.position.x;
    const double origin_y = info.origin.position.y;
    const double origin_z = info.origin.position.z;
    const size_t total_cells = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t data_len = msg->data.size();

    // classification counts
    size_t unknown_count = 0, free_count = 0, occupied_count = 0, dir_encoded_count = 0;
    int min_encoded = 256, max_encoded = -256; // will hold min/max of encoded (0..99) and occupied (100)
    const size_t SAMPLE_N = 20;
    std::vector<int> sample;
    sample.reserve(std::min(SAMPLE_N, data_len));

    for (size_t i = 0; i < data_len; ++i)
    {
      // msg->data holds int8_t values (-1..100); cast to int to inspect without char surprises
      int v = static_cast<int>(msg->data[i]);

      if (i < SAMPLE_N)
      {
        sample.push_back(v);
      }

      if (v == -1)
      {
        ++unknown_count;
      }
      else if (v == 0)
      {
        ++free_count;
        min_encoded = std::min(min_encoded, v);
        max_encoded = std::max(max_encoded, v);
      }
      else if (v == 100)
      {
        ++occupied_count;
        min_encoded = std::min(min_encoded, v);
        max_encoded = std::max(max_encoded, v);
      }
      else if (v >= 1 && v <= 99)
      {
        ++dir_encoded_count;
        min_encoded = std::min(min_encoded, v);
        max_encoded = std::max(max_encoded, v);
      }
      else
      {
        // Unexpected value (e.g., >100). Count as 'other' in debug output
        min_encoded = std::min(min_encoded, v);
        max_encoded = std::max(max_encoded, v);
      }
    }

    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] frame_id: " << msg->header.frame_id
                                                            << "  stamp(sec.nanosec): " << msg->header.stamp.sec << "." << msg->header.stamp.nanosec);
    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] width: " << width << ", height: " << height
                                                         << ", resolution: " << resolution << " m/cell");
    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] origin.position (x,y,z): "
                                    << origin_x << ", " << origin_y << ", " << origin_z);
    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] total_cells (width*height): " << total_cells
                                                                              << "  data.size(): " << data_len);

    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] cell counts -> unknown: " << unknown_count
                                                                          << ", free(0): " << free_count
                                                                          << ", occupied(100): " << occupied_count
                                                                          << ", direction_encoded(1..99): " << dir_encoded_count);

    if (min_encoded <= 255 && max_encoded >= -255)
    {
      RCLCPP_INFO_STREAM(_logger, "[DirectionMap] encoded range (min..max): " << min_encoded << " .. " << max_encoded);
    }

    // print a small sample for quick inspection
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < sample.size(); ++i)
    {
      if (i)
        ss << ", ";
      ss << sample[i];
    }
    ss << "]";
    RCLCPP_INFO_STREAM(_logger, "[DirectionMap] first " << sample.size() << " cells: " << ss.str());

    // sanity check: warn if sizes mismatch
    if (total_cells != data_len)
    {
      RCLCPP_WARN_STREAM(_logger, "[DirectionMap] total_cells (" << total_cells
                                                                 << ") != data.size() (" << data_len << ") -- map publisher may be malformed");
    }
  }

  if (!heading_marker_pub_)
  {
    RCLCPP_WARN(_logger, "heading_marker_pub_ is null!");
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  const auto now = _clock->now();

  const unsigned int W = msg->info.width;
  const unsigned int H = msg->info.height;
  const float res = msg->info.resolution;
  const float ox = msg->info.origin.position.x;
  const float oy = msg->info.origin.position.y;
  const std::string &frame_id = msg->header.frame_id;

  int marker_id = 0;
  unsigned int step = 32; // Only draw every 4th cell for speed

  for (unsigned int y = 0; y < H; y += step)
  {
    for (unsigned int x = 0; x < W; x += step)
    {
      unsigned int idx = y * W + x;
      int v = static_cast<int>(msg->data[idx]);
      int u = (v + 256) % 256;

      if (u < 1 || u > 99)
      {
        continue; // skip if not a direction-encoded cell
      }

      float wx = ox + (x + 0.5f) * res;
      float wy = oy + (y + 0.5f) * res;
      float theta = (u / 100.0f) * 2.0f * M_PI;

      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = now;
      m.ns = "direction_map_arrows";
      m.id = marker_id++;
      m.type = visualization_msgs::msg::Marker::ARROW;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = wx;
      m.pose.position.y = wy;
      m.pose.position.z = 0.05;

      // Convert heading theta to quaternion
      tf2::Quaternion q;
      q.setRPY(0, 0, theta);
      m.pose.orientation = tf2::toMsg(q);

      m.scale.x = 0.8;  // shaft length
      m.scale.y = 0.08; // shaft diameter
      m.scale.z = 0.08; // head diameter

      // Color: greenish
      m.color.r = 0.2f;
      m.color.g = 1.0f;
      m.color.b = 0.2f;
      m.color.a = 1.0f;

      m.lifetime = rclcpp::Duration::from_seconds(0.0); // persist
      marker_array.markers.push_back(m);
    }
  }

  heading_marker_pub_->publish(marker_array);
  RCLCPP_INFO(_logger, "[DirectionMap] Published %zu heading markers", marker_array.markers.size());
}
  nav_msgs::msg::Path SmacPlannerHybrid::createPlan(
      const geometry_msgs::msg::PoseStamped &start,
      const geometry_msgs::msg::PoseStamped &goal)
  {
    std::lock_guard<std::mutex> lock_reinit(_mutex);
    steady_clock::time_point a = steady_clock::now();

    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

    // Downsample costmap, if required
    nav2_costmap_2d::Costmap2D *costmap = _costmap;
    if (_costmap_downsampler)
    {
      costmap = _costmap_downsampler->downsample(_downsampling_factor);
      _collision_checker.setCostmap(costmap);
    }

    // Set collision checker and costmap information
    _collision_checker.setFootprint(
        _costmap_ros->getRobotFootprint(),
        _costmap_ros->getUseRadius(),
        findCircumscribedCost(_costmap_ros));
    _a_star->setCollisionChecker(&_collision_checker);

    // Set starting point, in A* bin search coordinates
    unsigned int mx, my;
    if (!costmap->worldToMap(start.pose.position.x, start.pose.position.y, mx, my))
    {
      throw std::runtime_error("Start pose is out of costmap!");
    }

    double orientation_bin = std::round(tf2::getYaw(start.pose.orientation) / _angle_bin_size);
    while (orientation_bin < 0.0)
    {
      orientation_bin += static_cast<float>(_angle_quantizations);
    }
    // This is needed to handle precision issues
    if (orientation_bin >= static_cast<float>(_angle_quantizations))
    {
      orientation_bin -= static_cast<float>(_angle_quantizations);
    }
    _a_star->setStart(mx, my, static_cast<unsigned int>(orientation_bin));

    // Set goal point, in A* bin search coordinates
    if (!costmap->worldToMap(goal.pose.position.x, goal.pose.position.y, mx, my))
    {
      throw std::runtime_error("Goal pose is out of costmap!");
    }
    orientation_bin = std::round(tf2::getYaw(goal.pose.orientation) / _angle_bin_size);
    while (orientation_bin < 0.0)
    {
      orientation_bin += static_cast<float>(_angle_quantizations);
    }
    // This is needed to handle precision issues
    if (orientation_bin >= static_cast<float>(_angle_quantizations))
    {
      orientation_bin -= static_cast<float>(_angle_quantizations);
    }
    _a_star->setGoal(mx, my, static_cast<unsigned int>(orientation_bin));
    // After successfully mapping goal pose into mx, my
    // _search_info.goal_x = static_cast<float>(mx);
    // _search_info.goal_y = static_cast<float>(my);

    // Setup message
    nav_msgs::msg::Path plan;
    plan.header.stamp = _clock->now();
    plan.header.frame_id = _global_frame;
    geometry_msgs::msg::PoseStamped pose;
    pose.header = plan.header;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = 0.0;
    pose.pose.orientation.w = 1.0;

    // Compute plan Hbrid A*
    NodeHybrid::CoordinateVector path;
    int num_iterations = 0;
    std::string error;
    try
    {
      if (!_a_star->createPath(
              path, num_iterations, _tolerance / static_cast<float>(costmap->getResolution())))
      {
        if (num_iterations < _a_star->getMaxIterations())
        {
          error = std::string("no valid path found");
        } else {
        error = std::string("exceeded maximum iterations");
      }
    }
  } catch (const std::runtime_error & e) {
    error = "invalid use: ";
    error += e.what();
  }

  if (!error.empty())
  {
    steady_clock::time_point b = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(b - a);
    RCLCPP_WARN(
        _logger,
        "%s: failed to create plan, %s.",
        _name.c_str(), error.c_str());

    RCLCPP_WARN(
        _logger,
        "%s: failed to create plan (%s) after %.3f sec and %d iterations",
        _name.c_str(), error.c_str(),
        time_span.count(), num_iterations);

    // Print profiling summary for failed plans
    RCLCPP_INFO(
        _logger,
        "[Planner Summary] Expanded Nodes: %zu | Checked Neighbors: %zu | "
        "Accepted: %zu | Rejected Heading: %zu | Rejected Collision: %zu",
        NodeHybrid::nodes_expanded,
        NodeHybrid::neighbors_checked,
        NodeHybrid::neighbors_accepted,
        NodeHybrid::neighbors_rejected_heading,
        NodeHybrid::neighbors_rejected_collision);

    // Reset profiling counters for the next planning attempt
    NodeHybrid::nodes_expanded = 0;
    NodeHybrid::neighbors_checked = 0;
    NodeHybrid::neighbors_accepted = 0;
    NodeHybrid::neighbors_rejected_heading = 0;
    NodeHybrid::neighbors_rejected_collision = 0;
    return plan;
  }

  // Convert to world coordinates
  plan.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.pose = getWorldCoords(path[i].x, path[i].y, costmap);
    pose.pose.orientation = getWorldOrientation(path[i].theta);
    plan.poses.push_back(pose);
  }

  // Publish raw path for debug
  if (_raw_plan_publisher->get_subscription_count() > 0) {
    _raw_plan_publisher->publish(plan);
  }

  // Find how much time we have left to do smoothing
  steady_clock::time_point b = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(b - a);
  double time_remaining = _max_planning_time - static_cast<double>(time_span.count());

  // Print profiling summary for successful plans
  RCLCPP_INFO(
      _logger,
      "[Planner Summary] Expanded Nodes: %zu | Checked Neighbors: %zu | "
      "Accepted: %zu | Rejected Heading: %zu | Rejected Collision: %zu",
      NodeHybrid::nodes_expanded,
      NodeHybrid::neighbors_checked,
      NodeHybrid::neighbors_accepted,
      NodeHybrid::neighbors_rejected_heading,
      NodeHybrid::neighbors_rejected_collision);

  RCLCPP_INFO(
    _logger,
    "%s: Successfully created plan in %.3f sec with %d iterations and %zu poses",
    _name.c_str(), time_span.count(), num_iterations, plan.poses.size());

  // Reset profiling counters after each plan
  NodeHybrid::nodes_expanded = 0;
  NodeHybrid::neighbors_checked = 0;
  NodeHybrid::neighbors_accepted = 0;
  NodeHybrid::neighbors_rejected_heading = 0;
  NodeHybrid::neighbors_rejected_collision = 0;


#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  // Smooth plan
  if (_smoother && num_iterations > 1) {
    _smoother->smooth(plan, costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  return plan;
}

rcl_interfaces::msg::SetParametersResult
SmacPlannerHybrid::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  std::lock_guard<std::mutex> lock_reinit(_mutex);

  bool reinit_collision_checker = false;
  bool reinit_a_star = false;
  bool reinit_downsampler = false;
  bool reinit_smoother = false;

  for (auto parameter : parameters)
  {
    const auto &type = parameter.get_type();
    const auto &name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE)
    {
      if (name == _name + ".max_planning_time")
      {
        reinit_a_star = true;
        _max_planning_time = parameter.as_double();
      }
      else if (name == _name + ".tolerance")
      {
        _tolerance = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".lookup_table_size")
      {
        reinit_a_star = true;
        _lookup_table_size = parameter.as_double();
      }
      else if (name == _name + ".minimum_turning_radius")
      {
        reinit_a_star = true;
        if (_smoother)
        {
          reinit_smoother = true;
        }
        _minimum_turning_radius_global_coords = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".reverse_penalty")
      {
        reinit_a_star = true;
        _search_info.reverse_penalty = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".change_penalty")
      {
        reinit_a_star = true;
        _search_info.change_penalty = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".non_straight_penalty")
      {
        reinit_a_star = true;
        _search_info.non_straight_penalty = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".cost_penalty")
      {
        reinit_a_star = true;
        _search_info.cost_penalty = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".analytic_expansion_ratio")
      {
        reinit_a_star = true;
        _search_info.analytic_expansion_ratio = static_cast<float>(parameter.as_double());
      }
      else if (name == _name + ".analytic_expansion_max_length")
      {
        reinit_a_star = true;
        _search_info.analytic_expansion_max_length =
            static_cast<float>(parameter.as_double()) / _costmap->getResolution();
        // --- Direction-map knobs (live-tunable doubles) ---
      }
      // else if (name == _name + ".direction_heading_decay")
      // {
      //   direction_heading_decay_ = parameter.as_double();
      //   _search_info.direction_heading_decay = direction_heading_decay_;
      // }
      else if (name == _name + ".direction_heading_weight")
      {
        direction_heading_weight_ = parameter.as_double();
        _search_info.direction_heading_weight = direction_heading_weight_;
      }
      // else if (name == _name + ".max_heading_deviation_rad")
      // {
      //  max_heading_deviation_rad_ = parameter.as_double();
      // _search_info.max_heading_deviation_rad = max_heading_deviation_rad_;
      // }
    }
    else if (type == ParameterType::PARAMETER_BOOL)
    {
      if (name == _name + ".downsample_costmap")
      {
        reinit_downsampler = true;
        _downsample_costmap = parameter.as_bool();
      }
      else if (name == _name + ".allow_unknown")
      {
        reinit_a_star = true;
        _allow_unknown = parameter.as_bool();
      }
      else if (name == _name + ".cache_obstacle_heuristic")
      {
        reinit_a_star = true;
        _search_info.cache_obstacle_heuristic = parameter.as_bool();
      }
      else if (name == _name + ".smooth_path")
      {
        if (parameter.as_bool())
        {
          reinit_smoother = true;
        }
        else
        {
          _smoother.reset();
        }
      }
      else if (name == _name + ".use_direction_map")
      {
        use_direction_map_ = parameter.as_bool();
        _search_info.use_direction_map = use_direction_map_;

        auto node = _node.lock();
        if (use_direction_map_)
        {
          if (!direction_map_)
            direction_map_ = std::make_shared<nav2_smac_planner::DirectionMap>();
          _search_info.direction_map = direction_map_;
          direction_map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
              direction_map_topic_, rclcpp::QoS(1).transient_local().reliable(),
              std::bind(&SmacPlannerHybrid::directionMapCallback, this, std::placeholders::_1));

          heading_marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
              "direction_map_vectors", rclcpp::QoS(1).transient_local());
        }
        else
        {
          direction_map_sub_.reset();
          _search_info.direction_map.reset();
        }
      }
    }
    else if (type == ParameterType::PARAMETER_INTEGER)
    {
      if (name == _name + ".downsampling_factor")
      {
        reinit_a_star = true;
        reinit_downsampler = true;
        _downsampling_factor = parameter.as_int();
      }
      else if (name == _name + ".max_iterations")
      {
        reinit_a_star = true;
        _max_iterations = parameter.as_int();
        if (_max_iterations <= 0)
        {
          RCLCPP_INFO(
              _logger, "maximum iteration selected as <= 0, "
                       "disabling maximum iterations.");
          _max_iterations = std::numeric_limits<int>::max();
        }
      }
      else if (name == _name + ".max_on_approach_iterations")
      {
        reinit_a_star = true;
        _max_on_approach_iterations = parameter.as_int();
        if (_max_on_approach_iterations <= 0)
        {
          RCLCPP_INFO(
              _logger, "On approach iteration selected as <= 0, "
                       "disabling tolerance and on approach iterations.");
          _max_on_approach_iterations = std::numeric_limits<int>::max();
        }
      }
      else if (name == _name + ".angle_quantization_bins")
      {
        reinit_collision_checker = true;
        reinit_a_star = true;
        int angle_quantizations = parameter.as_int();
        _angle_bin_size = 2.0 * M_PI / angle_quantizations;
        _angle_quantizations = static_cast<unsigned int>(angle_quantizations);
      }
    }
    else if (type == ParameterType::PARAMETER_STRING)
    {
      if (name == _name + ".motion_model_for_search")
      {
        reinit_a_star = true;
        _motion_model = fromString(parameter.as_string());
        if (_motion_model == MotionModel::UNKNOWN)
        {
          RCLCPP_WARN(
              _logger,
              "Unable to get MotionModel search type. Given '%s', "
              "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP.",
              _motion_model_for_search.c_str());
        }
        // --- Change direction-map topic on the fly ---
      }
      else if (name == _name + ".direction_map_topic")
      {
        direction_map_topic_ = parameter.as_string();
        if (use_direction_map_)
        {
          auto node = _node.lock();
          direction_map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
              direction_map_topic_, rclcpp::QoS(1).transient_local().reliable(),
              std::bind(&SmacPlannerHybrid::directionMapCallback, this, std::placeholders::_1));
        }
      }
    }
  }

  // Re-init if needed with mutex lock (to avoid re-init while creating a plan)
  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother) {
    // convert to grid coordinates
    if (!_downsample_costmap) {
      _downsampling_factor = 1;
    }
    _search_info.minimum_turning_radius =
      _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
    _lookup_table_dim =
      static_cast<float>(_lookup_table_size) /
      static_cast<float>(_costmap->getResolution() * _downsampling_factor);

    // Make sure its a whole number
    _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

    // Make sure its an odd number
    if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
      RCLCPP_INFO(
        _logger,
        "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
        _lookup_table_dim);
      _lookup_table_dim += 1.0;
    }

    auto node = _node.lock();

    // Re-Initialize A* template
    if (reinit_a_star) {
      _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
      _a_star->initialize(
        _allow_unknown,
        _max_iterations,
        _max_on_approach_iterations,
        _max_planning_time,
        _lookup_table_dim,
        _angle_quantizations);
    }

    // Re-Initialize costmap downsampler
    if (reinit_downsampler) {
      if (_downsample_costmap && _downsampling_factor > 1) {
        std::string topic_name = "downsampled_costmap";
        _costmap_downsampler = std::make_unique<CostmapDownsampler>();
        _costmap_downsampler->on_configure(
          node, _global_frame, topic_name, _costmap, _downsampling_factor);
      }
    }

    // Re-Initialize collision checker
    if (reinit_collision_checker) {
      _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations, node);
      _collision_checker.setFootprint(
        _costmap_ros->getRobotFootprint(),
        _costmap_ros->getUseRadius(),
        findCircumscribedCost(_costmap_ros));
    }

    // Re-Initialize smoother
    if (reinit_smoother) {
      SmootherParams params;
      params.get(node, _name);
      _smoother = std::make_unique<Smoother>(params);
      _smoother->initialize(_minimum_turning_radius_global_coords);
    }
  }
  result.successful = true;
  return result;
}

} // namespace nav2_smac_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_smac_planner::SmacPlannerHybrid, nav2_core::GlobalPlanner)
