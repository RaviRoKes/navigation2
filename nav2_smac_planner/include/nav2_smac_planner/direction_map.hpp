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