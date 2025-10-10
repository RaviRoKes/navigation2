//smac_planner_hybrid.cpp::

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
#include <unordered_map>
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

  //Direction-map guidance parameters
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
  if (heading_marker_pub_) heading_marker_pub_->on_activate();
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
  if (heading_marker_pub_) heading_marker_pub_->on_deactivate();
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
}

void SmacPlannerHybrid::directionMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  RCLCPP_INFO(_logger, "[DirectionMap] Directionmap callback ....Received map with size: %u x %u",
              msg->info.width, msg->info.height);

  // Prepare downsampled message (if we will do downsampling)
  const unsigned int factor = static_cast<unsigned int>(_downsampling_factor);
  const bool do_downsample = use_direction_map_ && _downsample_costmap && (factor > 1);

  nav_msgs::msg::OccupancyGrid ds_msg; // empty grid to hold downsampled map

  if (do_downsample) {
    // compute downsampled sizes
    const unsigned int W = msg->info.width;
    const unsigned int H = msg->info.height;
    const unsigned int newW = static_cast<unsigned int>(std::ceil(static_cast<float>(W) / static_cast<float>(factor)));
    const unsigned int newH = static_cast<unsigned int>(std::ceil(static_cast<float>(H) / static_cast<float>(factor)));

    // copy original and modify fields we need to change
    ds_msg = *msg;
    ds_msg.info.width = newW;
    ds_msg.info.height = newH;
    ds_msg.info.resolution = msg->info.resolution * static_cast<double>(factor);
    ds_msg.data.assign(newW * newH, static_cast<int8_t>(-1));

    // helpers for indexing
    auto in_index = [&](unsigned int ix, unsigned int iy) { return iy * W + ix; };
    auto out_index = [&](unsigned int ox, unsigned int oy) { return oy * newW + ox; };


    // Loop through each cell in the downsampled (output) map
    for (unsigned int oy = 0; oy < newH; ++oy) {
      for (unsigned int ox = 0; ox < newW; ++ox) {
        int obstacle_count = 0;
        int free_count = 0;
        bool has_direction = false ;

        unsigned int x_start = ox * factor;
        unsigned int y_start = oy * factor;
        // Check the corresponding block in the original to detect obstacle or free cells
        for (unsigned int by = 0; by < factor; ++by) {
          unsigned int iy = y_start + by;
          if (iy >= H) break;

          for (unsigned int bx = 0; bx < factor; ++bx) {
            unsigned int ix = x_start + bx;
            if (ix >= W) break;

            int raw_value = static_cast<int>(msg->data[in_index(ix, iy)]);
            if (raw_value == 100) {
              ++obstacle_count;
            } else if (raw_value >= 1 && raw_value <= 99) {
              has_direction = true;
            } else if (raw_value == 0) {
              ++free_count;
            } else {
              // unknown (-1) -> ignored unless nothing else present
            }
          }
        }
        // Decide what value to assign to this downsampled cell
        int8_t output_raw_value = -1;
        if (obstacle_count > 0)
        {
          output_raw_value = 100;
        }
        else
        {
          //Use center cell value
          unsigned int center_ix = x_start + factor / 2;
          unsigned int center_iy = y_start + factor / 2;

          // Clamp to map bounds
          if (center_ix >= W)
            center_ix = W - 1;
          if (center_iy >= H)
            center_iy = H - 1;

          int center_val = static_cast<int>(msg->data[in_index(center_ix, center_iy)]);

          if (center_val == 100)
          {
            output_raw_value = 100;
          }
          else if (center_val >= 1 && center_val <= 99)
          {
            output_raw_value = static_cast<int8_t>(center_val);
          }
          else if (center_val == 0)
          {
            output_raw_value = 0;
          }
          else
          {
            output_raw_value = -1;
          }
        }
        //Store in downsampled map
        ds_msg.data[out_index(ox, oy)] = output_raw_value;
      }
    }

    // set header/time/frame as source
    ds_msg.header = msg->header;
    direction_map_->setGrid(ds_msg);
    RCLCPP_INFO(_logger, "[DirectionMap] downsampled direction map to %u x %u, res=%.3f",
                ds_msg.info.width, ds_msg.info.height, ds_msg.info.resolution);
  } else {
    // no downsample: use original
    direction_map_->setGrid(*msg);
  }

  // At this point direction_map_ has been set (either with ds_msg or original)
  // Check alignment with planner active grid (resolution & origin)
  {
    const auto &used_info = do_downsample ? ds_msg.info : msg->info;

    const double dir_res = used_info.resolution;
    const double dir_ox = used_info.origin.position.x;
    const double dir_oy = used_info.origin.position.y;

    // planner active grid resolution (costmap resolution times downsampling factor used by planner)
    const double planner_res = _costmap->getResolution() * static_cast<double>(_downsampling_factor);
    const double planner_ox = _costmap->getOriginX();
    const double planner_oy = _costmap->getOriginY();

    const double eps = 1e-6;
    if (std::fabs(dir_res - planner_res) < eps &&
        std::fabs(dir_ox - planner_ox) < eps &&
        std::fabs(dir_oy - planner_oy) < eps)
    {
      _search_info.direction_map_aligned = true;
      RCLCPP_INFO(_logger, "[DirectionMap] aligned with planner grid (res %.6f m/cell)", dir_res);
    } else {
      _search_info.direction_map_aligned = false;
      RCLCPP_WARN(_logger,
        "[DirectionMap] NOT aligned: dir_res=%.6f origin=(%.3f,%.3f) planner_res=%.6f origin=(%.3f,%.3f)",
        dir_res, dir_ox, dir_oy, planner_res, planner_ox, planner_oy);
    }

    // ensure SearchInfo pointer updated
    _search_info.direction_map = direction_map_;
  }

  // Marker visualization: use the used message (downsampled if used)
  if (!heading_marker_pub_)
  {
    RCLCPP_WARN(_logger, "heading_marker_pub_ is null!");
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  const auto now = _clock->now();

  // choose which occupancy grid to read for visualization
  const nav_msgs::msg::OccupancyGrid *used_grid = (do_downsample ? &ds_msg : msg.get());

  const unsigned int W = used_grid->info.width;
  const unsigned int H = used_grid->info.height;
  const float res = static_cast<float>(used_grid->info.resolution);
  const float ox = static_cast<float>(used_grid->info.origin.position.x);
  const float oy = static_cast<float>(used_grid->info.origin.position.y);
  const std::string &frame_id = used_grid->header.frame_id;

  int marker_id = 0;
  unsigned int step = 16; // draw every N cells to keep marker count reasonable

  for (unsigned int y = 0; y < H; y += step)
  {
    for (unsigned int x = 0; x < W; x += step)
    {
      unsigned int idx = y * W + x;
      int v = static_cast<int>(used_grid->data[idx]);
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

      m.scale.x = 0.8f;  // shaft length
      m.scale.y = 0.08f; // shaft diameter
      m.scale.z = 0.08f; // head diameter

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
  nav2_costmap_2d::Costmap2D * costmap = _costmap;
  if (_costmap_downsampler) {
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
  if (!costmap->worldToMap(start.pose.position.x, start.pose.position.y, mx, my)) {
    throw std::runtime_error("Start pose is out of costmap!");
  }

  double orientation_bin = std::round(tf2::getYaw(start.pose.orientation) / _angle_bin_size);
  while (orientation_bin < 0.0) {
    orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (orientation_bin >= static_cast<float>(_angle_quantizations)) {
    orientation_bin -= static_cast<float>(_angle_quantizations);
  }
  _a_star->setStart(mx, my, static_cast<unsigned int>(orientation_bin));

  // Set goal point, in A* bin search coordinates
  if (!costmap->worldToMap(goal.pose.position.x, goal.pose.position.y, mx, my)) {
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

  // Compute plan
  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  std::string error;
  try {
    if (!_a_star->createPath(
            path, num_iterations, _tolerance / static_cast<float>(costmap->getResolution())))
    {
      if (num_iterations < _a_star->getMaxIterations())
      {
        error = std::string("no valid path found");
      }
      else {
        error = std::string("exceeded maximum iterations");
      }
    }
  }
  catch (const std::runtime_error &e)
  {
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
  for (int i = path.size() - 1; i >= 0; --i)
  {
    pose.pose = getWorldCoords(path[i].x, path[i].y, costmap);
    pose.pose.orientation = getWorldOrientation(path[i].theta);
    plan.poses.push_back(pose);
  }

  // Publish raw path for debug
  if (_raw_plan_publisher->get_subscription_count() > 0)
  {
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
  std::cout << "It took " << time_span.count() * 1000 << " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  // Smooth plan
  if (_smoother && num_iterations > 1)
  {
    _smoother->smooth(plan, costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 << " milliseconds to smooth path." << std::endl;
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
        // --- Direction-map
      }
      else if (name == _name + ".direction_heading_weight")
      {
        direction_heading_weight_ = parameter.as_double();
        _search_info.direction_heading_weight = direction_heading_weight_;
      }
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
  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother)
  {
    // convert to grid coordinates
    if (!_downsample_costmap)
    {
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
    if (static_cast<int>(_lookup_table_dim) % 2 == 0)
    {
      RCLCPP_INFO(
          _logger,
          "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
          _lookup_table_dim);
      _lookup_table_dim += 1.0;
    }

    auto node = _node.lock();

    // Re-Initialize A* template
    if (reinit_a_star)
    {
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
    if (reinit_downsampler)
    {
      if (_downsample_costmap && _downsampling_factor > 1)
      {
        std::string topic_name = "downsampled_costmap";
        _costmap_downsampler = std::make_unique<CostmapDownsampler>();
        _costmap_downsampler->on_configure(
            node, _global_frame, topic_name, _costmap, _downsampling_factor);
      }
    }

    // Re-Initialize collision checker
    if (reinit_collision_checker)
    {
      _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations, node);
      _collision_checker.setFootprint(
          _costmap_ros->getRobotFootprint(),
          _costmap_ros->getUseRadius(),
          findCircumscribedCost(_costmap_ros));
    }

    // Re-Initialize smoother
    if (reinit_smoother)
    {
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



//node_hybrid.cpp::

// Copyright (c) 2020, Samsung Research America
// Copyright (c) 2020, Applied Electric Vehicles Pty Ltd
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

#include <math.h>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>
#include <queue>
#include <limits>
#include <utility>
#include <cmath>
#include "angles/angles.h"

#include "ompl/base/ScopedState.h"
#include "ompl/base/spaces/DubinsStateSpace.h"
#include "ompl/base/spaces/ReedsSheppStateSpace.h"

#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/direction_map.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono;  // NOLINT

namespace nav2_smac_planner
{

// defining static member for all instance to share
LookupTable NodeHybrid::obstacle_heuristic_lookup_table;
double NodeHybrid::travel_distance_cost = sqrt(2);
HybridMotionTable NodeHybrid::motion_table;
float NodeHybrid::size_lookup = 25;
LookupTable NodeHybrid::dist_heuristic_lookup_table;
nav2_costmap_2d::Costmap2D * NodeHybrid::sampled_costmap = nullptr;
CostmapDownsampler NodeHybrid::downsampler;
ObstacleHeuristicQueue NodeHybrid::obstacle_heuristic_queue;
SearchInfo* NodeHybrid::search_info = nullptr;

// --- Profiling counters ---
size_t NodeHybrid::neighbors_checked = 0;
size_t NodeHybrid::neighbors_accepted = 0;
size_t NodeHybrid::neighbors_rejected_heading = 0;
size_t NodeHybrid::neighbors_rejected_collision = 0;
size_t NodeHybrid::nodes_expanded = 0;

// Each of these tables are the projected motion models through
// time and space applied to the search on the current node in
// continuous map-coordinates (e.g. not meters but partial map cells)
// Currently, these are set to project *at minimum* into a neighboring
// cell. Though this could be later modified to project a certain
// amount of time or particular distance forward.

// http://planning.cs.uiuc.edu/node821.html
// Model for ackermann style vehicle with minimum radius restriction
void HybridMotionTable::initDubin(
  unsigned int & size_x_in,
  unsigned int & /*size_y_in*/,
  unsigned int & num_angle_quantization_in,
  SearchInfo & search_info)
{
  size_x = size_x_in;
  change_penalty = search_info.change_penalty;
  non_straight_penalty = search_info.non_straight_penalty;
  cost_penalty = search_info.cost_penalty;
  reverse_penalty = search_info.reverse_penalty;
  travel_distance_reward = 1.0f - search_info.retrospective_penalty;

  // if nothing changed, no need to re-compute primitives
  if (num_angle_quantization_in == num_angle_quantization &&
    min_turning_radius == search_info.minimum_turning_radius &&
    motion_model == MotionModel::DUBIN)
  {
    return;
  }

  num_angle_quantization = num_angle_quantization_in;
  num_angle_quantization_float = static_cast<float>(num_angle_quantization);
  min_turning_radius = search_info.minimum_turning_radius;
  motion_model = MotionModel::DUBIN;

  // angle must meet 3 requirements:
  // 1) be increment of quantized bin size
  // 2) chord length must be greater than sqrt(2) to leave current cell
  // 3) maximum curvature must be respected, represented by minimum turning angle
  // Thusly:
  // On circle of radius minimum turning angle, we need select motion primatives
  // with chord length > sqrt(2) and be an increment of our bin size
  //
  // chord >= sqrt(2) >= 2 * R * sin (angle / 2); where angle / N = quantized bin size
  // Thusly: angle <= 2.0 * asin(sqrt(2) / (2 * R))
  float angle = 2.0 * asin(sqrt(2.0) / (2 * min_turning_radius));
  // Now make sure angle is an increment of the quantized bin size
  // And since its based on the minimum chord, we need to make sure its always larger
  bin_size =
    2.0f * static_cast<float>(M_PI) / static_cast<float>(num_angle_quantization);
  float increments;
  if (angle < bin_size) {
    increments = 1.0f;
  } else {
    // Search dimensions are clean multiples of quantization - this prevents
    // paths with loops in them
    increments = ceil(angle / bin_size);
  }
  angle = increments * bin_size;

  // find deflections
  // If we make a right triangle out of the chord in circle of radius
  // min turning angle, we can see that delta X = R * sin (angle)
  float delta_x = min_turning_radius * sin(angle);
  // Using that same right triangle, we can see that the complement
  // to delta Y is R * cos (angle). If we subtract R, we get the actual value
  float delta_y = min_turning_radius - (min_turning_radius * cos(angle));

  projections.clear();
  projections.reserve(3);
  projections.emplace_back(hypotf(delta_x, delta_y), 0.0, 0.0);  // Forward
  projections.emplace_back(delta_x, delta_y, increments);  // Left
  projections.emplace_back(delta_x, -delta_y, -increments);  // Right

  // Create the correct OMPL state space
  state_space = std::make_unique<ompl::base::DubinsStateSpace>(min_turning_radius);

  // Precompute projection deltas
  delta_xs.resize(projections.size());
  delta_ys.resize(projections.size());
  trig_values.resize(num_angle_quantization);

  for (unsigned int i = 0; i != projections.size(); i++) {
    delta_xs[i].resize(num_angle_quantization);
    delta_ys[i].resize(num_angle_quantization);

    for (unsigned int j = 0; j != num_angle_quantization; j++) {
      double cos_theta = cos(bin_size * j);
      double sin_theta = sin(bin_size * j);
      if (i == 0) {
        // if first iteration, cache the trig values for later
        trig_values[j] = {cos_theta, sin_theta};
      }
      delta_xs[i][j] = projections[i]._x * cos_theta - projections[i]._y * sin_theta;
      delta_ys[i][j] = projections[i]._x * sin_theta + projections[i]._y * cos_theta;
    }
  }
}

// http://planning.cs.uiuc.edu/node822.html
// Same as Dubin model but now reverse is valid
// See notes in Dubin for explanation
void HybridMotionTable::initReedsShepp(
  unsigned int & size_x_in,
  unsigned int & /*size_y_in*/,
  unsigned int & num_angle_quantization_in,
  SearchInfo & search_info)
{
  size_x = size_x_in;
  change_penalty = search_info.change_penalty;
  non_straight_penalty = search_info.non_straight_penalty;
  cost_penalty = search_info.cost_penalty;
  reverse_penalty = search_info.reverse_penalty;
  travel_distance_reward = 1.0f - search_info.retrospective_penalty;

  // if nothing changed, no need to re-compute primitives
  if (num_angle_quantization_in == num_angle_quantization &&
    min_turning_radius == search_info.minimum_turning_radius &&
    motion_model == MotionModel::REEDS_SHEPP)
  {
    return;
  }

  num_angle_quantization = num_angle_quantization_in;
  num_angle_quantization_float = static_cast<float>(num_angle_quantization);
  min_turning_radius = search_info.minimum_turning_radius;
  motion_model = MotionModel::REEDS_SHEPP;

  float angle = 2.0 * asin(sqrt(2.0) / (2 * min_turning_radius));
  bin_size =
    2.0f * static_cast<float>(M_PI) / static_cast<float>(num_angle_quantization);
  float increments;
  if (angle < bin_size) {
    increments = 1.0f;
  } else {
    increments = ceil(angle / bin_size);
  }
  angle = increments * bin_size;

  float delta_x = min_turning_radius * sin(angle);
  float delta_y = min_turning_radius - (min_turning_radius * cos(angle));

  projections.clear();
  projections.reserve(6);
  projections.emplace_back(hypotf(delta_x, delta_y), 0.0, 0.0);  // Forward
  projections.emplace_back(delta_x, delta_y, increments);  // Forward + Left
  projections.emplace_back(delta_x, -delta_y, -increments);  // Forward + Right
  projections.emplace_back(-hypotf(delta_x, delta_y), 0.0, 0.0);  // Backward
  projections.emplace_back(-delta_x, delta_y, -increments);  // Backward + Left
  projections.emplace_back(-delta_x, -delta_y, increments);  // Backward + Right

  // Create the correct OMPL state space
  state_space = std::make_unique<ompl::base::ReedsSheppStateSpace>(min_turning_radius);

  // Precompute projection deltas
  delta_xs.resize(projections.size());
  delta_ys.resize(projections.size());
  trig_values.resize(num_angle_quantization);

  for (unsigned int i = 0; i != projections.size(); i++) {
    delta_xs[i].resize(num_angle_quantization);
    delta_ys[i].resize(num_angle_quantization);

    for (unsigned int j = 0; j != num_angle_quantization; j++) {
      double cos_theta = cos(bin_size * j);
      double sin_theta = sin(bin_size * j);
      if (i == 0) {
        // if first iteration, cache the trig values for later
        trig_values[j] = {cos_theta, sin_theta};
      }
      delta_xs[i][j] = projections[i]._x * cos_theta - projections[i]._y * sin_theta;
      delta_ys[i][j] = projections[i]._x * sin_theta + projections[i]._y * cos_theta;
    }
  }
}

MotionPoses HybridMotionTable::getProjections(const NodeHybrid * node)
{
  MotionPoses projection_list;
  projection_list.reserve(projections.size());

  for (unsigned int i = 0; i != projections.size(); i++) {
    const MotionPose & motion_model = projections[i];

    // normalize theta, I know its overkill, but I've been burned before...
    const float & node_heading = node->pose.theta;
    float new_heading = node_heading + motion_model._theta;

    if (new_heading < 0.0) {
      new_heading += num_angle_quantization_float;
    }

    if (new_heading >= num_angle_quantization_float) {
      new_heading -= num_angle_quantization_float;
    }

    projection_list.emplace_back(
      delta_xs[i][node_heading] + node->pose.x,
      delta_ys[i][node_heading] + node->pose.y,
      new_heading);
  }

  return projection_list;
}

unsigned int HybridMotionTable::getClosestAngularBin(const double & theta)
{
  auto bin = static_cast<unsigned int>(round(static_cast<float>(theta) / bin_size));
  return bin < num_angle_quantization ? bin : 0u;
}

float HybridMotionTable::getAngleFromBin(const unsigned int & bin_idx)
{
  return bin_idx * bin_size;
}

NodeHybrid::NodeHybrid(const unsigned int index)
: parent(nullptr),
  pose(0.0f, 0.0f, 0.0f),
  _cell_cost(std::numeric_limits<float>::quiet_NaN()),
  _accumulated_cost(std::numeric_limits<float>::max()),
  _index(index),
  _was_visited(false),
  _motion_primitive_index(std::numeric_limits<unsigned int>::max())
{
}

NodeHybrid::~NodeHybrid()
{
  parent = nullptr;
}

void NodeHybrid::reset()
{
  parent = nullptr;
  _cell_cost = std::numeric_limits<float>::quiet_NaN();
  _accumulated_cost = std::numeric_limits<float>::max();
  _was_visited = false;
  _motion_primitive_index = std::numeric_limits<unsigned int>::max();
  pose.x = 0.0f;
  pose.y = 0.0f;
  pose.theta = 0.0f;
}

bool NodeHybrid::isNodeValid(
  const bool & traverse_unknown,
  GridCollisionChecker * collision_checker)
{
  if (collision_checker->inCollision(
      this->pose.x, this->pose.y, this->pose.theta /*bin number*/, traverse_unknown))
  {
    return false;
  }

  _cell_cost = collision_checker->getCost();
  return true;
}

float NodeHybrid::getTraversalCost(const NodePtr &child)
{
  const float normalized_cost = child->getCost() / 252.0;
  if (std::isnan(normalized_cost))
  {
    throw std::runtime_error(
        "Node attempted to get traversal "
        "cost without a known SE2 collision cost!");
  }

  // this is the first node
  if (getMotionPrimitiveIndex() == std::numeric_limits<unsigned int>::max())
  {
    return NodeHybrid::travel_distance_cost;
  }

  float travel_cost = 0.0;
  float travel_cost_raw =
      NodeHybrid::travel_distance_cost *
      (motion_table.travel_distance_reward + motion_table.cost_penalty * normalized_cost);

  if (child->getMotionPrimitiveIndex() == 0 || child->getMotionPrimitiveIndex() == 3)
  {
    // New motion is a straight motion, no additional costs to be applied
    travel_cost = travel_cost_raw;
  }
  else
  {
    if (getMotionPrimitiveIndex() == child->getMotionPrimitiveIndex())
    {
      // Turning motion but keeps in same direction: encourages to commit to turning if starting it
      travel_cost = travel_cost_raw * motion_table.non_straight_penalty;
    }
    else
    {
      // Turning motion and changing direction: penalizes wiggling
      travel_cost = travel_cost_raw *
                    (motion_table.non_straight_penalty + motion_table.change_penalty);
    }
  }

  if (child->getMotionPrimitiveIndex() > 2)
  {
    // reverse direction
    travel_cost *= motion_table.reverse_penalty;
  }
  // Direction map integration
  {
    auto si = NodeHybrid::search_info;
    // auto dir_map = (si) ? si->direction_map : nullptr;
    if (si == nullptr)
    { /* no search info */
    }
    else
    {
      const float W = static_cast<float>(si->direction_heading_weight);
      if (si->use_direction_map && W > 0.0f && si->direction_map && si->direction_map->isValid())
      {
        auto dir_map = si->direction_map; // shared_ptr copy -> keeps it alive during use
        // child pose to nearest map indices
        const int mx = static_cast<int>(std::lround(child->pose.x));
        const int my = static_cast<int>(std::lround(child->pose.y));
        // bounds check before calling into map
        if (mx>= 0 && my>= 0 &&
            static_cast<unsigned int>(mx) < dir_map->getSizeInCellsX() &&
            static_cast<unsigned int>(my) < dir_map->getSizeInCellsY())
        {
          ++NodeHybrid::neighbors_checked;

          float theta_ref = 0.0f; // preferred heading (radians) from direction map
          if (dir_map->getPreferredTheta(static_cast<unsigned int>(mx),
                                         static_cast<unsigned int>(my),
                                         theta_ref))
          {

            // Compute the actual motion heading
            const float dx = child->pose.x - this->pose.x;
            const float dy = child->pose.y - this->pose.y;
            // skip degenerate zero-length motion
            if (!(dx == 0.0f && dy == 0.0f))
            {
              const float theta_motion = std::atan2(dy, dx);
              // minimal angular difference [-π, π]: (theta_motion - theta_ref)
              const float dtheta = static_cast<float>(
                  angles::shortest_angular_distance(static_cast<double>(theta_ref),
                                                    static_cast<double>(theta_motion)));

              const float norm = std::fabs(dtheta) / static_cast<float>(M_PI); // [0,1]
              const float heading_factor = 1.0f + W * (norm * norm);

              travel_cost *= heading_factor;
              ++NodeHybrid::neighbors_accepted;
              // RCLCPP_DEBUG(
              //     rclcpp::get_logger("smac_planner"),
              //     "dir-wt: mx=%d my=%d ref=%.3f motion=%.3f dtheta=%.3f norm=%.3f W=%.3f factor=%.3f",
              //     mx, my, theta_ref, theta_motion, dtheta, norm, W, heading_factor);
            }
            else
            {
              // degenerate motion (rare)
              ++NodeHybrid::neighbors_rejected_heading;
            }
          }
          else
          {
            // no preferred heading for this cell (occupied/unknown/zero)
            ++NodeHybrid::neighbors_rejected_heading;
          }
        }
      }
    }
  }

  return travel_cost;
}

float NodeHybrid::getHeuristicCost(
    const Coordinates &node_coords,
    const Coordinates &goal_coords,
    const nav2_costmap_2d::Costmap2D * /*costmap*/)
{
  const float obstacle_heuristic =
      getObstacleHeuristic(node_coords, goal_coords, motion_table.cost_penalty);
  const float dist_heuristic = getDistanceHeuristic(node_coords, goal_coords, obstacle_heuristic);
  return std::max(obstacle_heuristic, dist_heuristic);
}

void NodeHybrid::initMotionModel(
    const MotionModel &motion_model,
    unsigned int &size_x,
    unsigned int &size_y,
    unsigned int &num_angle_quantization,
    SearchInfo &search_info)
{
  // find the motion model selected
  switch (motion_model)
  {
  case MotionModel::DUBIN:
    motion_table.initDubin(size_x, size_y, num_angle_quantization, search_info);
    break;
  case MotionModel::REEDS_SHEPP:
    motion_table.initReedsShepp(size_x, size_y, num_angle_quantization, search_info);
    break;
  default:
    throw std::runtime_error(
        "Invalid motion model for Hybrid A*. Please select between"
        " Dubin (Ackermann forward only),"
        " Reeds-Shepp (Ackermann forward and back).");
  }

  travel_distance_cost = motion_table.projections[0]._x;
}

inline float distanceHeuristic2D(
    const unsigned int idx, const unsigned int size_x,
    const unsigned int target_x, const unsigned int target_y)
{
  int dx = static_cast<int>(idx % size_x) - static_cast<int>(target_x);
  int dy = static_cast<int>(idx / size_x) - static_cast<int>(target_y);
  return std::sqrt(dx * dx + dy * dy);
}

void NodeHybrid::resetObstacleHeuristic(
    nav2_costmap_2d::Costmap2D *costmap,
    const unsigned int &start_x, const unsigned int &start_y,
    const unsigned int &goal_x, const unsigned int &goal_y)
{
  // Downsample costmap 2x to compute a sparse obstacle heuristic. This speeds up
  // the planner considerably to search through 75% less cells with no detectable
  // erosion of path quality after even modest smoothing. The error would be no more
  // than 0.05 * normalized cost. Since this is just a search prior, there's no loss in generality
  std::weak_ptr<nav2_util::LifecycleNode> ptr;
  downsampler.on_configure(ptr, "fake_frame", "fake_topic", costmap, 2.0, true);
  downsampler.on_activate();
  sampled_costmap = downsampler.downsample(2.0);

  // Clear lookup table
  unsigned int size = sampled_costmap->getSizeInCellsX() * sampled_costmap->getSizeInCellsY();
  if (obstacle_heuristic_lookup_table.size() == size){
    // must reset all values
    std::fill(
        obstacle_heuristic_lookup_table.begin(),
        obstacle_heuristic_lookup_table.end(), 0.0);
  }else{
    unsigned int obstacle_size = obstacle_heuristic_lookup_table.size();
    obstacle_heuristic_lookup_table.resize(size, 0.0);
    // must reset values for non-constructed indices
    std::fill_n(
        obstacle_heuristic_lookup_table.begin(), obstacle_size, 0.0);
  }

  obstacle_heuristic_queue.clear();
  obstacle_heuristic_queue.reserve(
      sampled_costmap->getSizeInCellsX() * sampled_costmap->getSizeInCellsY());

  // Set initial goal point to queue from. Divided by 2 due to downsampled costmap.
  const unsigned int size_x = sampled_costmap->getSizeInCellsX();
  const unsigned int goal_index = floor(goal_y / 2.0) * size_x + floor(goal_x / 2.0);
  obstacle_heuristic_queue.emplace_back(
      distanceHeuristic2D(goal_index, size_x, start_x, start_y), goal_index);

  // initialize goal cell with a very small value to differentiate it from 0.0 (~uninitialized)
  // the negative value means the cell is in the open set
  obstacle_heuristic_lookup_table[goal_index] = -0.00001f;
}

float NodeHybrid::getObstacleHeuristic(
    const Coordinates &node_coords,
    const Coordinates &goal_coords,
    const double &cost_penalty)
{
  // If already expanded, return the cost
  const unsigned int size_x = sampled_costmap->getSizeInCellsX();
  // Divided by 2 due to downsampled costmap.
  const unsigned int start_y = floor(node_coords.y / 2.0);
  const unsigned int start_x = floor(node_coords.x / 2.0);
  const unsigned int start_index = start_y * size_x + start_x;
  const float &requested_node_cost = obstacle_heuristic_lookup_table[start_index];
  if (requested_node_cost > 0.0f)
  {
    // costs are doubled due to downsampling
    return 2.0 * requested_node_cost;
  }

  // If not, expand until it is included. This dynamic programming ensures that
  // we only expand the MINIMUM spanning set of the costmap per planning request.
  // Rather than naively expanding the entire (potentially massive) map for a limited
  // path, we only expand to the extent required for the furthest expansion in the
  // search-planning request that dynamically updates during search as needed.

  // start_x and start_y have changed since last call
  // we need to recompute 2D distance heuristic and reprioritize queue
  for (auto &n : obstacle_heuristic_queue){
    n.first = -obstacle_heuristic_lookup_table[n.second] +
              distanceHeuristic2D(n.second, size_x, start_x, start_y);
  }
  std::make_heap(
      obstacle_heuristic_queue.begin(), obstacle_heuristic_queue.end(),
      ObstacleHeuristicComparator{});

  const int size_x_int = static_cast<int>(size_x);
  const unsigned int size_y = sampled_costmap->getSizeInCellsY();
  const float sqrt_2 = sqrt(2);
  float c_cost, cost, travel_cost, new_cost, existing_cost;
  unsigned int idx, mx, my, mx_idx, my_idx;
  unsigned int new_idx = 0;

  const std::vector<int> neighborhood = {1, -1,                             // left right
                                         size_x_int, -size_x_int,           // up down
                                         size_x_int + 1, size_x_int - 1,    // upper diagonals
                                         -size_x_int + 1, -size_x_int - 1}; // lower diagonals

  while (!obstacle_heuristic_queue.empty()){
    idx = obstacle_heuristic_queue.front().second;
    std::pop_heap(
        obstacle_heuristic_queue.begin(), obstacle_heuristic_queue.end(),
        ObstacleHeuristicComparator{});
    obstacle_heuristic_queue.pop_back();
    c_cost = obstacle_heuristic_lookup_table[idx];
    if (c_cost > 0.0f){
      // cell has been processed and closed, no further cost improvements
      // are mathematically possible thanks to euclidean distance heuristic consistency
      continue;
    }
    c_cost = -c_cost;
    obstacle_heuristic_lookup_table[idx] = c_cost; // set a positive value to close the cell

    my_idx = idx / size_x;
    mx_idx = idx - (my_idx * size_x);

    // find neighbors
    for (unsigned int i = 0; i != neighborhood.size(); i++){
      new_idx = static_cast<unsigned int>(static_cast<int>(idx) + neighborhood[i]);

      // if neighbor path is better and non-lethal, set new cost and add to queue
      if (new_idx < size_x * size_y){
        cost = static_cast<float>(sampled_costmap->getCost(new_idx));
        if (cost >= INSCRIBED){
          continue;
        }

        my = new_idx / size_x;
        mx = new_idx - (my * size_x);

        if (mx == 0 && mx_idx >= size_x - 1 || mx >= size_x - 1 && mx_idx == 0){
          continue;
        }
        if (my == 0 && my_idx >= size_y - 1 || my >= size_y - 1 && my_idx == 0){
          continue;
        }

        existing_cost = obstacle_heuristic_lookup_table[new_idx];
        if (existing_cost <= 0.0f){
          travel_cost =
              ((i <= 3) ? 1.0f : sqrt_2) * (1.0f + (cost_penalty * cost / 252.0f));
          new_cost = c_cost + travel_cost;
          if (existing_cost == 0.0f || -existing_cost > new_cost)
          {
            // the negative value means the cell is in the open set
            obstacle_heuristic_lookup_table[new_idx] = -new_cost;
            obstacle_heuristic_queue.emplace_back(
                new_cost + distanceHeuristic2D(new_idx, size_x, start_x, start_y), new_idx);
            std::push_heap(
                obstacle_heuristic_queue.begin(), obstacle_heuristic_queue.end(),
                ObstacleHeuristicComparator{});
          }
        }
      }
    }

    if (idx == start_index){
      break;
    }
  }

  // return requested_node_cost which has been updated by the search
  // costs are doubled due to downsampling
  return 2.0 * requested_node_cost;
}

float NodeHybrid::getDistanceHeuristic(
    const Coordinates &node_coords,
    const Coordinates &goal_coords,
    const float &obstacle_heuristic)
{
  // rotate and translate node_coords such that goal_coords relative is (0,0,0)
  // Due to the rounding involved in exact cell increments for caching,
  // this is not an exact replica of a live heuristic, but has bounded error.
  // (Usually less than 1 cell)

  // This angle is negative since we are de-rotating the current node
  // by the goal angle; cos(-th) = cos(th) & sin(-th) = -sin(th)
  const TrigValues &trig_vals = motion_table.trig_values[goal_coords.theta];
  const float cos_th = trig_vals.first;
  const float sin_th = -trig_vals.second;
  const float dx = node_coords.x - goal_coords.x;
  const float dy = node_coords.y - goal_coords.y;

  double dtheta_bin = node_coords.theta - goal_coords.theta;
  if (dtheta_bin < 0){
    dtheta_bin += motion_table.num_angle_quantization;
  }
  if (dtheta_bin > motion_table.num_angle_quantization){
    dtheta_bin -= motion_table.num_angle_quantization;
  }

  Coordinates node_coords_relative(
      round(dx * cos_th - dy * sin_th),
      round(dx * sin_th + dy * cos_th),
      round(dtheta_bin));

  // Check if the relative node coordinate is within the localized window around the goal
  // to apply the distance heuristic. Since the lookup table is contains only the positive
  // X axis, we mirror the Y and theta values across the X axis to find the heuristic values.
  float motion_heuristic = 0.0;
  const int floored_size = floor(size_lookup / 2.0);
  const int ceiling_size = ceil(size_lookup / 2.0);
  const float mirrored_relative_y = abs(node_coords_relative.y);
  if (abs(node_coords_relative.x) < floored_size && mirrored_relative_y < floored_size)
  {
    // Need to mirror angle if Y coordinate was mirrored
    int theta_pos;
    if (node_coords_relative.y < 0.0){
      theta_pos = motion_table.num_angle_quantization - node_coords_relative.theta;
    }else{
      theta_pos = node_coords_relative.theta;
    }
    const int x_pos = node_coords_relative.x + floored_size;
    const int y_pos = static_cast<int>(mirrored_relative_y);
    const int index =
        x_pos * ceiling_size * motion_table.num_angle_quantization +
        y_pos * motion_table.num_angle_quantization +
        theta_pos;
    motion_heuristic = dist_heuristic_lookup_table[index];
  }
  else if (obstacle_heuristic <= 0.0)
  {
    // If no obstacle heuristic value, must have some H to use
    // In nominal situations, this should never be called.
    static ompl::base::ScopedState<> from(motion_table.state_space), to(motion_table.state_space);
    to[0] = goal_coords.x;
    to[1] = goal_coords.y;
    to[2] = goal_coords.theta * motion_table.num_angle_quantization;
    from[0] = node_coords.x;
    from[1] = node_coords.y;
    from[2] = node_coords.theta * motion_table.num_angle_quantization;
    motion_heuristic = motion_table.state_space->distance(from(), to());
  }

  return motion_heuristic;
}

void NodeHybrid::precomputeDistanceHeuristic(
    const float & lookup_table_dim,
    const MotionModel & motion_model,
    const unsigned int & dim_3_size,
    const SearchInfo & search_info)
{
  // Dubin or Reeds-Shepp shortest distances
  if (motion_model == MotionModel::DUBIN) {
    motion_table.state_space = std::make_unique<ompl::base::DubinsStateSpace>(
        search_info.minimum_turning_radius);
  } else if (motion_model == MotionModel::REEDS_SHEPP) {
    motion_table.state_space = std::make_unique<ompl::base::ReedsSheppStateSpace>(
        search_info.minimum_turning_radius);
  } else {
    throw std::runtime_error(
        "Node attempted to precompute distance heuristics "
        "with invalid motion model!");
  }

  ompl::base::ScopedState<> from(motion_table.state_space), to(motion_table.state_space);
  to[0] = 0.0;
  to[1] = 0.0;
  to[2] = 0.0;
  size_lookup = lookup_table_dim;
  float motion_heuristic = 0.0;
  unsigned int index = 0;
  int dim_3_size_int = static_cast<int>(dim_3_size);
  float angular_bin_size = 2 * M_PI / static_cast<float>(dim_3_size);

  // Create a lookup table of Dubin/Reeds-Shepp distances in a window around the goal
  // to help drive the search towards admissible approaches. Deu to symmetries in the
  // Heuristic space, we need to only store 2 of the 4 quadrants and simply mirror
  // around the X axis any relative node lookup. This reduces memory overhead and increases
  // the size of a window a platform can store in memory.
  dist_heuristic_lookup_table.resize(size_lookup * ceil(size_lookup / 2.0) * dim_3_size_int);
  for (float x = ceil(-size_lookup / 2.0); x <= floor(size_lookup / 2.0); x += 1.0) {
    for (float y = 0.0; y <= floor(size_lookup / 2.0); y += 1.0) {
      for (int heading = 0; heading != dim_3_size_int; heading++) {
        from[0] = x;
        from[1] = y;
        from[2] = heading * angular_bin_size;
        motion_heuristic = motion_table.state_space->distance(from(), to());
        dist_heuristic_lookup_table[index] = motion_heuristic;
        index++;
      }
    }
  }
}
void NodeHybrid::getNeighbors(
    std::function<bool(const unsigned int &, nav2_smac_planner::NodeHybrid *&)> & NeighborGetter,
    GridCollisionChecker *collision_checker,
    const bool & traverse_unknown,
    NodeVector & neighbors)
{
  unsigned int index = 0;
  NodePtr neighbor = nullptr;
  Coordinates initial_node_coords;
  const MotionPoses motion_projections = motion_table.getProjections(this);

  for (unsigned int i = 0; i != motion_projections.size(); i++) {
    index = NodeHybrid::getIndex(
        static_cast<unsigned int>(motion_projections[i]._x),
        static_cast<unsigned int>(motion_projections[i]._y),
        static_cast<unsigned int>(motion_projections[i]._theta),
        motion_table.size_x, motion_table.num_angle_quantization);

    if (NeighborGetter(index, neighbor) && !neighbor->wasVisited()) {
      // Cache the initial pose in case it was visited but valid
      // don't want to disrupt continuous coordinate expansion
      initial_node_coords = neighbor->pose;
      neighbor->setPose(
          Coordinates(
              motion_projections[i]._x,
              motion_projections[i]._y,
              motion_projections[i]._theta));
      if (neighbor->isNodeValid(traverse_unknown, collision_checker)) {
        neighbor->setMotionPrimitiveIndex(i);
        neighbors.push_back(neighbor);
      } else {
        neighbor->setPose(initial_node_coords);
      }
    }
  }
}

bool NodeHybrid::backtracePath(CoordinateVector & path)
{
  if (!this->parent) {
    return false;
  }

  NodePtr current_node = this;

  while (current_node->parent) {
    path.push_back(current_node->pose);
    // Convert angle to radians
    path.back().theta = NodeHybrid::motion_table.getAngleFromBin(path.back().theta);
    current_node = current_node->parent;
  }

  // add the start pose
  path.push_back(current_node->pose);
  // Convert angle to radians
  path.back().theta = NodeHybrid::motion_table.getAngleFromBin(path.back().theta);

  return true;
}

} // namespace nav2_smac_planner


///direction_map.hpp::
#pragma once

#include <vector>
#include <limits>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <string>
#include <cstdint>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace nav2_smac_planner
{
  class DirectionMap
  {
  public:
    using OG = nav_msgs::msg::OccupancyGrid;
    void setGrid(const OG &msg)
    {
      std::lock_guard<std::mutex> lk(m_);
      size_x_ = msg.info.width;
      size_y_ = msg.info.height;
      resolution_ = msg.info.resolution; // meters per cell=, 0.05m/pixel)
      origin_x_ = msg.info.origin.position.x;
      origin_y_ = msg.info.origin.position.y;
      frame_ = msg.header.frame_id;
      data_ = msg.data; // int8_t, -1 means unknown
      have_ = (size_x_ > 0 && size_y_ > 0 && resolution_ > 0.0);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "DirectionMap Origin: (%.2f, %.2f)", origin_x_, origin_y_);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "Direction map loaded: %ux%u, res: %.3f m/cell",
                  size_x_, size_y_, resolution_);
    }
    // check if map is usable
    bool isValid() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return have_;
    }
    // Getters (Costmap2D style)
    unsigned int getSizeInCellsX() const { return size_x_; }
    unsigned int getSizeInCellsY() const { return size_y_; }
    double getResolution() const { return resolution_; }
    double getOriginX() const { return origin_x_; }
    double getOriginY() const { return origin_y_; }
    double getSizeInMetersX() const { return static_cast<double>(size_x_) * resolution_; }
    double getSizeInMetersY() const { return static_cast<double>(size_y_) * resolution_; }

    // Conversion: map <-> world (same as Costmap2D)
    void mapToWorld(unsigned int mx, unsigned int my, double &wx, double &wy) const
    {
      std::lock_guard<std::mutex> lk(m_);
      wx = origin_x_ + (mx + 0.5) * resolution_;
      wy = origin_y_ + (my + 0.5) * resolution_;
    }

    // World wx, wy (meters) -> map indices,, mx = colun index x
    bool worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_ || wx < origin_x_ || wy < origin_y_)
        return false;
      mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
      my = static_cast<unsigned int>((wy - origin_y_) / resolution_);
      return mx < size_x_ && my < size_y_;
    }
    // Enforce bounds version (clamps mx, my to map edges) ---
    void worldToMapEnforceBounds(double wx, double wy, int &mx, int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (wx < origin_x_)
      {
        mx = 0;
      }
      else if (wx > origin_x_ + resolution_ * size_x_)
      {
        mx = size_x_ - 1;
      }
      else
      {
        mx = static_cast<int>((wx - origin_x_) / resolution_);
      }
      if (wy < origin_y_)
      {
        my = 0;
      }
      else if (wy > origin_y_ + resolution_ * size_y_)
      {
        my = size_y_ - 1;
      }
      else
      {
        my = static_cast<int>((wy - origin_y_) / resolution_);
      }
    }

    // Utility: check if map indices are inside map bounds ---
    bool isInsideMap(unsigned int mx, unsigned int my) const
    {
      return mx < size_x_ && my < size_y_;
    }

    // return radians in [0, 2π), heading at cell , index
    bool getPreferredTheta(unsigned int mx, unsigned int my, float &theta) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_ || mx >= size_x_ || my >= size_y_)
      {
        RCLCPP_WARN(rclcpp::get_logger("direction_map"),
                    "Coordinates out of bounds or map invalid");
        return false;
      }

      unsigned int idx = my * size_x_ + mx;
      int raw = static_cast<int>(data_[idx]);
      int wrapped = (raw + 256) % 256;

      RCLCPP_DEBUG(rclcpp::get_logger("direction_map"),
                   "getPreferredTheta: (%u,%u) → raw=%d → wrapped=%d",
                   mx, my, raw, wrapped);

      if (wrapped == 0 || wrapped > 99)
      {
        return false; // invalid or no preferred heading
      }

      theta = static_cast<float>(wrapped) / 100.0f * 2.0f * static_cast<float>(M_PI);
      return true;
    }
    bool getPreferredThetaWorld(double wx, double wy, float &theta) const
    {
      unsigned int mx, my;
      if (!worldToMap(wx, wy, mx, my))
      {
        RCLCPP_WARN(rclcpp::get_logger("direction_map"),
                    "[getPreferredThetaWorld] worldToMap failed for (%.2f, %.2f)", wx, wy);
        return false;
      }

      return getPreferredTheta(mx, my, theta);
    }

    std::string frameId() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return frame_;
    }

    const std::vector<int8_t> &getData() const { return data_; }

  private:
    mutable std::mutex m_;
    bool have_{false};
    unsigned int size_x_{0}, size_y_{0};
    double resolution_{0.0}, origin_x_{0.0}, origin_y_{0.0};
    std::string frame_;
    std::vector<int8_t> data_;
  };

} // namespace nav2_smac_planner