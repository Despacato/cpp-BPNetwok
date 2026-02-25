/**
 * @author Despacato
 * @date 2026/2/20
 * @Email dlmu_zxg@163.com
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include "Net.h"

namespace Utils{
  static double sigmoid(double x){
    return 1.0/(1.0 + std::exp(-x));
  }

  std::vector<double> getFileData(const std::string& filename);
  std::vector<Sample> getTrainData(const std::string& filename);
  std::vector<Sample> getTestData(const std::string& filename);

}
