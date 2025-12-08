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

    // Basic setter (no downsampling) - preserves existing behaviour
    void setGrid(const OG &msg)
    {
      std::lock_guard<std::mutex> lk(m_);
      size_x_ = msg.info.width;
      size_y_ = msg.info.height;
      resolution_ = msg.info.resolution; // meters per cell
      origin_x_ = msg.info.origin.position.x;
      origin_y_ = msg.info.origin.position.y;
      frame_ = msg.header.frame_id;
      data_ = msg.data; // int8_t, -1 means unknown
      downsampling_factor_ = 1;
      have_ = (size_x_ > 0 && size_y_ > 0 && resolution_ > 0.0);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "DirectionMap Origin: (%.2f, %.2f)", origin_x_, origin_y_);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "Direction map loaded: %ux%u, res: %.3f m/cell",
                  size_x_, size_y_, resolution_);
    }

    // Downsampling setter — uses center-cell of each block (no mean/mode)
    // factor: integer downsampling factor (1 = no downsample)
    void setGrid(const OG &msg, int factor)
    {
      if (factor <= 1)
      {
        // behave like the simple setter
        setGrid(msg);
        return;
      }

      std::lock_guard<std::mutex> lk(m_);

      const unsigned int W = msg.info.width;
      const unsigned int H = msg.info.height;
      const unsigned int newW = static_cast<unsigned int>(std::ceil(static_cast<float>(W) / static_cast<float>(factor)));
      const unsigned int newH = static_cast<unsigned int>(std::ceil(static_cast<float>(H) / static_cast<float>(factor)));

      std::vector<int8_t> ds_data;
      ds_data.assign(newW * newH, static_cast<int8_t>(-1));

      // helpers for indexing
      auto in_index = [&](unsigned int ix, unsigned int iy) { return iy * W + ix; };
      auto out_index = [&](unsigned int ox, unsigned int oy) { return oy * newW + ox; };

      for (unsigned int oy = 0; oy < newH; ++oy)
      {
        for (unsigned int ox = 0; ox < newW; ++ox)
        {
          int obstacle_count = 0;
          int free_count = 0;

          unsigned int x_start = ox * factor;
          unsigned int y_start = oy * factor;

          for (unsigned int by = 0; by < static_cast<unsigned int>(factor); ++by)
          {
            unsigned int iy = y_start + by;
            if (iy >= H) break;

            for (unsigned int bx = 0; bx < static_cast<unsigned int>(factor); ++bx)
            {
              unsigned int ix = x_start + bx;
              if (ix >= W) break;

              int raw_value = static_cast<int>(msg.data[in_index(ix, iy)]);
              if (raw_value == 100)
              {
                ++obstacle_count;
              }
              else if (raw_value == 0)
              {
                ++free_count;
              }
              // other values (1..99) indicate direction cells; we don't aggregate them here
            }
          }

          int8_t output_raw_value = -1;

          if (obstacle_count > 0)
          {
            output_raw_value = 100; // obstacle-dominant policy
          }
          else
          {
            // center cell fallback (use the center cell of the block)
            unsigned int center_ix = x_start + factor / 2;
            unsigned int center_iy = y_start + factor / 2;

            if (center_ix >= W) center_ix = W - 1;
            if (center_iy >= H) center_iy = H - 1;

            int center_val = static_cast<int>(msg.data[in_index(center_ix, center_iy)]);

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

          ds_data[out_index(ox, oy)] = output_raw_value;
        }
      }

      // Assign downsized meta and data to this object
      size_x_ = newW;
      size_y_ = newH;
      resolution_ = msg.info.resolution * static_cast<double>(factor);
      origin_x_ = msg.info.origin.position.x;
      origin_y_ = msg.info.origin.position.y;
      frame_ = msg.header.frame_id;
      data_.swap(ds_data);
      downsampling_factor_ = factor;
      have_ = (size_x_ > 0 && size_y_ > 0 && resolution_ > 0.0);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "Direction map downsampled: %ux%u -> %ux%u, new res: %.3f m/cell",
                  W, H, size_x_, size_y_, resolution_);
    }

    // check if map is usable
    bool isValid() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return have_;
    }

    // Getters (Costmap2D style)
    unsigned int getSizeInCellsX() const { std::lock_guard<std::mutex> lk(m_); return size_x_; }
    unsigned int getSizeInCellsY() const { std::lock_guard<std::mutex> lk(m_); return size_y_; }
    double getResolution() const { std::lock_guard<std::mutex> lk(m_); return resolution_; }
    double getOriginX() const { std::lock_guard<std::mutex> lk(m_); return origin_x_; }
    double getOriginY() const { std::lock_guard<std::mutex> lk(m_); return origin_y_; }
    double getSizeInMetersX() const { std::lock_guard<std::mutex> lk(m_); return static_cast<double>(size_x_) * resolution_; }
    double getSizeInMetersY() const { std::lock_guard<std::mutex> lk(m_); return static_cast<double>(size_y_) * resolution_; }

    // Conversion: map <-> world (same as Costmap2D)
    void mapToWorld(unsigned int mx, unsigned int my, double &wx, double &wy) const
    {
      std::lock_guard<std::mutex> lk(m_);
      wx = origin_x_ + (mx + 0.5) * resolution_;
      wy = origin_y_ + (my + 0.5) * resolution_;
    }

    // World wx, wy (meters) -> map indices, mx = column index x
    bool worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_ || wx < origin_x_ || wy < origin_y_)
        return false;
      mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
      my = static_cast<unsigned int>((wy - origin_y_) / resolution_);
      return mx < size_x_ && my < size_y_;
    }

    // Enforce bounds version (clamps mx, my to map edges) --- keeps int refs to preserve caller semantics
    void worldToMapEnforceBounds(double wx, double wy, int &mx, int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      const double max_x = origin_x_ + resolution_ * static_cast<double>(size_x_);
      const double max_y = origin_y_ + resolution_ * static_cast<double>(size_y_);

      if (wx <= origin_x_)
      {
        mx = 0;
      }
      else if (wx >= max_x)
      {
        mx = static_cast<int>(size_x_) - 1;
      }
      else
      {
        mx = static_cast<int>((wx - origin_x_) / resolution_);
      }

      if (wy <= origin_y_)
      {
        my = 0;
      }
      else if (wy >= max_y)
      {
        my = static_cast<int>(size_y_) - 1;
      }
      else
      {
        my = static_cast<int>((wy - origin_y_) / resolution_);
      }
    }

    // check if map indices are inside map bounds --- thread-safe
    bool isInsideMap(unsigned int mx, unsigned int my) const
    {
      std::lock_guard<std::mutex> lk(m_);
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

      unsigned int idx = static_cast<unsigned int>(my) * size_x_ + mx;
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

    // Convenience: query preferred heading directly from world coords (meters)
    // This avoids double-locking (single lock inside this method) and clamps to nearest cell.
    bool getPreferredThetaAtWorld(double wx, double wy, float &theta) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_)
      {
        return false;
      }

      // clamp or compute indices (same logic as worldToMapEnforceBounds but done under same lock)
      const double max_x = origin_x_ + resolution_ * static_cast<double>(size_x_);
      const double max_y = origin_y_ + resolution_ * static_cast<double>(size_y_);

      int mx_i;
      int my_i;

      if (wx <= origin_x_)
        mx_i = 0;
      else if (wx >= max_x)
        mx_i = static_cast<int>(size_x_) - 1;
      else
        mx_i = static_cast<int>((wx - origin_x_) / resolution_);

      if (wy <= origin_y_)
        my_i = 0;
      else if (wy >= max_y)
        my_i = static_cast<int>(size_y_) - 1;
      else
        my_i = static_cast<int>((wy - origin_y_) / resolution_);

      if (mx_i < 0 || my_i < 0)
        return false;

      unsigned int mx = static_cast<unsigned int>(mx_i);
      unsigned int my = static_cast<unsigned int>(my_i);
      unsigned int idx = my * size_x_ + mx;
      int raw = static_cast<int>(data_[idx]);
      int wrapped = (raw + 256) % 256;

      if (wrapped == 0 || wrapped > 99)
        return false;

      theta = static_cast<float>(wrapped) / 100.0f * 2.0f * static_cast<float>(M_PI);
      return true;
    }

    std::string frameId() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return frame_;
    }

    const std::vector<int8_t> &getData() const { return data_; }
    // Downsampling factor support
    void setDownsamplingFactor(int factor)
    {
      std::lock_guard<std::mutex> lk(m_);
      downsampling_factor_ = factor;
    }

    int getDownsamplingFactor() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return downsampling_factor_;
    }

  private:
    mutable std::mutex m_;
    bool have_{false};
    unsigned int size_x_{0}, size_y_{0};
    double resolution_{0.0}, origin_x_{0.0}, origin_y_{0.0};
    std::string frame_;
    std::vector<int8_t> data_;
    int downsampling_factor_{1};
  };

} // namespace nav2_smac_planner
