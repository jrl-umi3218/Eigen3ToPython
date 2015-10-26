// This file is part of Eigen3ToPython.
// 
// Eigen3ToPython is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Eigen3ToPython is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with Eigen3ToPython.  If not, see <http://www.gnu.org/licenses/>.

#ifdef __clang__
#include <memory>
#endif

Eigen::Quaterniond* createFromAngleAxis(double angle, const Eigen::Vector3d& axis)
{
#ifdef __clang__
  std::allocator<Eigen::Quaterniond> alloc;
  Eigen::Quaterniond * ret = alloc.allocate(1);
  alloc.construct(ret, Eigen::AngleAxisd(angle, axis));
  return ret;
#else
  return new Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
#endif
}

Eigen::Quaterniond* createFromMatrix(const Eigen::Matrix3d& rot)
{
#ifdef __clang__
  std::allocator<Eigen::Quaterniond> alloc;
  Eigen::Quaterniond * ret = alloc.allocate(1);
  alloc.construct(ret, Eigen::AngleAxisd(rot));
  return ret;
#else
  return new Eigen::Quaterniond(Eigen::AngleAxisd(rot));
#endif
}

std::ostream& operator<<(std::ostream& out, const Eigen::Quaterniond& q)
{
  out << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z();
  return out;
}

