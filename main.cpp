/**
 * @author Despacato
 * @date 2026/2/20
 * @Email dlmu_zxg@163.com
 */

#include <iostream>
#include <string>
#include "lib/Net.h"
#include "lib/Utils.h"

using namespace std;

int main(int argc ,char* argv[]){
  // Create neural network object
  Net net;
  //Read training data
  const vector<Sample> trainDataSet = Utils::getTrainData("../data/traindata.txt");

  //Training neural network
  net.train(trainDataSet);
  //Prediction of samples using neural network
  const vector<Sample> testDataSet = Utils::getTestData("../data/testdata.txt");
  vector<Sample> predSet = net.predict(testDataSet);
  for(auto& pred: predSet){
    pred.display();
  }
  return 0;
}
