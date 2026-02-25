/**
 * @author Deapacato
 * @date 2026/2/20
 * @Email dlmu_zxg@163.com
 */

#pragma once

#include <vector>
#include "Config.h"

using std::vector;

struct Sample{
  vector<double> feature,label;

  Sample();

  Sample(const vector<double>& feature, const vector<double>& label);

  void display();
};

struct Node{
  double value{},bias{},bias_delta{};
  vector<double> weight,weight_delta;
  explicit Node(size_t nextLayerSize) : weight(nextLayerSize,0.f),weight_delta(nextLayerSize,0.f){}
};

class Net{
 private:
  Node *inputLayer[Config::INNODE]{};
  Node *hiddenLayer[Config::HIDENODE]{};
  Node *outputLayer[Config::OUTNODE]{};

  void grad_zero();

  void forward();

  double calculateLoss(const vector<double>& label);

  void backward(const vector<double>& label);

  void revise(size_t batch_size);
 public:
  Net();

  bool train(const vector<Sample>& trainDataSet);

  Sample predict(const vector<double>& feature);

  vector<Sample> predict(const vector<Sample>& predictDataSet);
};
