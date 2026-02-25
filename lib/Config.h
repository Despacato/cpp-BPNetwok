/**
 * @author Despacato
 * @date 2026/2/20
 * @Email dlmu_zxg@163.com
 */

#pragma once

using std::size_t;

namespace Config{
  const size_t INNODE = 2;
  const size_t HIDENODE = 4;
  const size_t OUTNODE = 1;

  const double lr = 0.1;
  const double threshold = 1e-4;
  const size_t max_epoch = 1e6;
}
