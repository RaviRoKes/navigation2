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

    DirectionMap() = default;

    /**
     * Set the raw occupancy-grid (can be high-res). The DirectionMap caches this
     * raw message and builds an internal downsampled representation using the
     * current downsampling_factor_.
     */
    void setGrid(const OG &msg)
    {
      std::lock_guard<std::mutex> lk(m_);
      last_raw_msg_ = msg; // cache raw message
      processRawGrid();    // build internal (possibly downsampled) grid
    }

    // Set/get downsampling factor (1 = no downsampling). Calling the setter
    // will reprocess any cached raw grid immediately.
    void setDownsamplingFactor(unsigned int factor)
    {
      if (factor < 1) factor = 1;
      {
        std::lock_guard<std::mutex> lk(m_);
        if (downsampling_factor_ == factor) return;
        downsampling_factor_ = factor;
      }
      // Reprocess inside lock
      std::lock_guard<std::mutex> lk(m_);
      if (!last_raw_msg_.data.empty()) {
        processRawGrid(); // rebuild internal grid with new factor
      }
    }

    unsigned int getDownsamplingFactor() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return downsampling_factor_;
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

    // World wx, wy (meters) -> map indices (mx = column index x)
    bool worldToMap(double wx, double wy, unsigned int &mx, unsigned int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_ || wx < origin_x_ || wy < origin_y_)
        return false;
      mx = static_cast<unsigned int>((wx - origin_x_) / resolution_);
      my = static_cast<unsigned int>((wy - origin_y_) / resolution_);
      return mx < size_x_ && my < size_y_;
    }

    // Enforce bounds version (clamps mx, my to map edges)
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

    // Utility: check if map indices are inside map bounds
    bool isInsideMap(unsigned int mx, unsigned int my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      return mx < size_x_ && my < size_y_;
    }

    // return radians in [0, 2π), heading at cell index. Returns false if none.
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
    // re-build internal grid from last_raw_msg_ using downsampling_factor_
    void processRawGrid()
    {
      // assume caller holds lock, but re-lock for safety
      std::lock_guard<std::mutex> lk(m_);
      if (last_raw_msg_.data.empty()) {
        have_ = false;
        return;
      }

      const unsigned int inW = last_raw_msg_.info.width;
      const unsigned int inH = last_raw_msg_.info.height;
      const double inRes = last_raw_msg_.info.resolution;
      const double inOx = last_raw_msg_.info.origin.position.x;
      const double inOy = last_raw_msg_.info.origin.position.y;

      if (downsampling_factor_ <= 1) {
        // use raw as-is
        size_x_ = inW;
        size_y_ = inH;
        resolution_ = inRes;
        origin_x_ = inOx;
        origin_y_ = inOy;
        frame_ = last_raw_msg_.header.frame_id;
        data_ = last_raw_msg_.data;
        have_ = (size_x_ > 0 && size_y_ > 0 && resolution_ > 0.0);
        RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                    "DirectionMap loaded raw: %ux%u, res: %.3f m/cell",
                    size_x_, size_y_, resolution_);
        return;
      }

      const unsigned int factor = downsampling_factor_;
      const unsigned int outW = static_cast<unsigned int>(std::ceil(static_cast<float>(inW) / static_cast<float>(factor)));
      const unsigned int outH = static_cast<unsigned int>(std::ceil(static_cast<float>(inH) / static_cast<float>(factor)));
      std::vector<int8_t> out_data(outW * outH, static_cast<int8_t>(-1));
      auto in_index = [&](unsigned int ix, unsigned int iy) { return iy * inW + ix; };
      auto out_index = [&](unsigned int ox, unsigned int oy) { return oy * outW + ox; };

      // For each output cell, pick center if it contains direction / obstacle / free,
      // otherwise fallback to circular mean of direction cells in block.
      for (unsigned int oy = 0; oy < outH; ++oy) {
        for (unsigned int ox = 0; ox < outW; ++ox) {
          unsigned int x_start = ox * factor;
          unsigned int y_start = oy * factor;

          int obstacle_count = 0;
          int free_count = 0;
          std::vector<int> dir_vals;
          // scan block
          for (unsigned int by = 0; by < factor; ++by) {
            unsigned int iy = y_start + by;
            if (iy >= inH) break;
            for (unsigned int bx = 0; bx < factor; ++bx) {
              unsigned int ix = x_start + bx;
              if (ix >= inW) break;
              int raw_value = static_cast<int>(last_raw_msg_.data[in_index(ix, iy)]);
              if (raw_value == 100) {
                ++obstacle_count;
              } else if (raw_value >= 1 && raw_value <= 99) {
                dir_vals.push_back(raw_value);
              } else if (raw_value == 0) {
                ++free_count;
              } else {
                // unknown (-1) -> ignore
              }
            }
          }

          int8_t out_val = -1;
          if (obstacle_count > 0) {
            out_val = 100;
          } else {
            // choose center cell
            unsigned int center_ix = x_start + ((factor - 1) / 2);
            unsigned int center_iy = y_start + ((factor - 1) / 2);
            if (center_ix >= inW) center_ix = inW - 1;
            if (center_iy >= inH) center_iy = inH - 1;
            int center_val = static_cast<int>(last_raw_msg_.data[in_index(center_ix, center_iy)]);
            if (center_val == 100) {
              out_val = 100;
            } else if (center_val >= 1 && center_val <= 99) {
              out_val = static_cast<int8_t>(center_val);
            } else if (!dir_vals.empty()) {
              // circular mean fallback
              double sum_s = 0.0, sum_c = 0.0;
              for (int dv : dir_vals) {
                double theta = (static_cast<double>(dv) / 100.0) * 2.0f * M_PI;
                sum_s += std::sin(theta);
                sum_c += std::cos(theta);
              }
              double mean_theta = std::atan2(sum_s, sum_c);
              if (mean_theta < 0.0) mean_theta += 2.0 * M_PI;
              int enc = static_cast<int>(std::round((mean_theta / (2.0 * M_PI)) * 100.0));
              if (enc <= 0) enc = 1;
              if (enc >= 100) enc = 99;
              out_val = static_cast<int8_t>(enc);
            } else if (free_count > 0) {
              out_val = 0;
            } else {
              out_val = -1;
            }
          }

          out_data[out_index(ox, oy)] = out_val;
        }
      }

      // set internal fields
      size_x_ = outW;
      size_y_ = outH;
      resolution_ = inRes * static_cast<double>(factor);
      origin_x_ = inOx;
      origin_y_ = inOy;
      frame_ = last_raw_msg_.header.frame_id;
      data_.swap(out_data);
      have_ = (size_x_ > 0 && size_y_ > 0 && resolution_ > 0.0);

      RCLCPP_INFO(rclcpp::get_logger("direction_map"),
                  "DirectionMap processed: %ux%u, res: %.3f m/cell, factor=%u",
                  size_x_, size_y_, resolution_, factor);
    }

  private:
    mutable std::mutex m_;
    bool have_{false};

    // internal (possibly downsampled) grid info
    unsigned int size_x_{0}, size_y_{0};
    double resolution_{0.0}, origin_x_{0.0}, origin_y_{0.0};
    std::string frame_;
    std::vector<int8_t> data_;

    // cached original raw grid (as received from topic) so we can reprocess with different factor
    OG last_raw_msg_{};

    // current downsampling factor (1 = raw)
    unsigned int downsampling_factor_{1};
  };

} // namespace nav2_smac_planner
