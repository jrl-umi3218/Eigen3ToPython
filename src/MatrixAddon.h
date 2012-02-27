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

inline Matrix(const Scalar& s0, const Scalar& s1, const Scalar& s2,
    const Scalar& s3, const Scalar& s4, const Scalar& s5)
{
  Base::_check_template_params();
  m_storage.data()[0] = s0;
  m_storage.data()[1] = s1;
  m_storage.data()[2] = s2;
  m_storage.data()[3] = s3;
  m_storage.data()[4] = s4;
  m_storage.data()[5] = s5;
}

