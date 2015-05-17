//
//  CWRecurrentNetwork.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__CWRecurrentNetwork__
#define __prediction__CWRecurrentNetwork__

#include <stdexcept>
#include <vector>
#include "ActivationFunctions.h"
#include "MemoryBlock.h"
#include "NeuralLayer.h"
#include "NeuralNetwork.h"

using std::vector;
using std::logic_error;

class CWRecurrentNetwork : public NeuralNetwork {
public:
	int step;
	
	int inputUnits;
	int outputUnits;
	int hiddenUnits;
	int modules;
	ActivationFunction* activationFunction;
	
	NeuralLayer* inputLayer;
	NeuralLayer* outputLayer;
	NeuralLayer* thresholdLayer;
	
	vector<NeuralLayer*> hiddenLayerModules;
	vector<NeuralLayer*> contextLayerModules;
	vector<int> modulesClockRate;
	
	CWRecurrentNetwork(int inputUnits, int hiddenModuleUnits, int outputUnits, const vector<int>& modulesClockRate, ActivationFunction* activationFunction);
	~CWRecurrentNetwork();
	
	void Propagate(const MemoryBlock& input);
};

#endif /* defined(__prediction__CWRecurrentNetwork__) */
