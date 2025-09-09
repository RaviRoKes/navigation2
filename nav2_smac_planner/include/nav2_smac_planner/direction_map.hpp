#pragma once

#include <vector>
#include <atomic>
#include <limits>
#include <cmath>
#include <mutex>
#include <algorithm>
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
      info_ = msg.info; // copy map meta data: size, resolu
      data_ = msg.data; // copy grid cell dtata
      frame_id_ = msg.header.frame_id;
      have_ = true; // set flag : have map now
    }
    // Quick validity check
    bool isValid() const
    {
      std::lock_guard<std::mutex> lk(m_);
      return have_ &&
             info_.width > 0 && info_.height > 0 &&
             info_.resolution > 0.0;
    }

    // World wx, wy (meters) -> map indices
    bool worldToMap(double wx, double wy, int &mx, int &my) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_)
        return false;

      const double res = info_.resolution; // meters per cell? (e.g., 0.05m/pixel)
      const double ox = info_.origin.position.x;
      const double oy = info_.origin.position.y;
      // Convert real-world coordinates (meters) to grid cell coordinate map (40x30)
      mx = static_cast<int>(std::floor((wx - ox) / res));
      my = static_cast<int>(std::floor((wy - oy) / res));
      // Check if the calculated cell is actually inside the map boundaries
      return mx >= 0 && my >= 0 &&
             mx < static_cast<int>(info_.width) &&
             my < static_cast<int>(info_.height);
    }

    // return radians in [0, 2π) if available; false = no preference
    bool getPreferredTheta(unsigned int mx, unsigned int my, float &theta_ref) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_ || mx >= info_.width || my >= info_.height)
        return false;
      // Calculate the index of the cell in the 1D data array
      const int idx = static_cast<int>(my) * static_cast<int>(info_.width) + static_cast<int>(mx);

      // Read signed and unsigned views
      const int8_t s = data_[idx];               // Get the cell value as a signed byte (-128 to 127)(ROS unknown = -1)
      const uint8_t u = static_cast<uint8_t>(s); // unsigned (0..255)

      // // unknown cell → no preference
      // if (s == -1)
      //   return false;

      // treat pure white as "no preference"
      // (so only painted corridor cells bias the heading)
      if (u >= 255)
        return false;

      // map value to [0, 1] depending on encoding
      float norm;
      if (u <= 100)
      {
        // mode: scale (map_server's 0..100)
        norm = static_cast<float>(u) / 100.0f;
      }
      else
      {
        // mode: raw (full 0..255)
        norm = static_cast<float>(u) / 255.0f;
      }
      // Convert the normalized value (0->1) to an angle in radians (0 -> 2*Pi)
      theta_ref = norm * 2.0f * static_cast<float>(M_PI);
      return true;
    }
    // Convenience: same as above but returns NaN when undefined (used by NodeHybrid)
    float lookupHeadingRad(int mx, int my) const
    {
      float theta;
      if (mx < 0 || my < 0)
        return std::numeric_limits<float>::quiet_NaN();
      if (getPreferredTheta(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), theta))
      {
        return theta;
      }
      return std::numeric_limits<float>::quiet_NaN();
    }

    // Very simple “distance” proxy in cells for decay:
    //  - how "far" a cell is from the nearest cell with a preferred direction.
    // Right now, it's very simple
    //  - it's either 0 ("right here!") or 50 ("somewhere else").
    float distanceCells(int mx, int my) const
    {
      float theta;
      if (mx < 0 || my < 0)
        return std::numeric_limits<float>::infinity();
      // Check if the cell has a defined direction
      if (getPreferredTheta(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), theta))
      {
        return 0.0f;
      }
      return 50.0f; // tune or replace with proper EDT/DT:ÖEuclidean Distance Transform
    }

    // quick metadata check
    bool metaMatches(float resolution, double origin_x, double origin_y) const
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!have_)
        return false;
      // Check if this map's properties (resolution, origin) match the given ones.
      // Used to see if a new map is actually different from the current one.
      return std::abs(info_.resolution - resolution) < 1e-6 &&
             std::abs(info_.origin.position.x - origin_x) < 1e-6 &&
             std::abs(info_.origin.position.y - origin_y) < 1e-6;
    }

  private:
    mutable std::mutex m_;
    nav_msgs::msg::MapMetaData info_; // Map metadata (size, resolution, origin)
    std::vector<int8_t> data_;        // actual grid data, a list of values for each cell.
    std::string frame_id_;
    bool have_{false};
  };

} // namespace nav2_smac_planner
