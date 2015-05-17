//
//  NeuralLayer.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__NeuralLayer__
#define __prediction__NeuralLayer__

#include <vector>
#include <stdexcept>
#include "ActivationFunctions.h"
#include "MemoryBlock.h"

using std::vector;
using std::logic_error;

class NeuralLayer {
public:
	int units;
	float momentumRate;
	float learningRate;
	
	ActivationFunction* activationFunction;
	
	MemoryBlock activation;
	MemoryBlock activationDerivative;
	MemoryBlock weights;
	MemoryBlock weightsDelta;
	MemoryBlock delta;
	
	int inputUnits;
	vector<int> inputWeightsOffsets;
	vector<int> outputWeightsOffsets;
	
	vector<NeuralLayer*> connectionsIn;
	vector<NeuralLayer*> connectionsOut;
	
	NeuralLayer(int units, ActivationFunction* activationFunction, float learningRate, float momentumRate);
	~NeuralLayer() {};
	
	void ProjectTo(NeuralLayer* nextLayer);
	
	void SetActivation(const MemoryBlock& input);
	void GetActivation(MemoryBlock& output);
	void SetTarget(const MemoryBlock& target);
	
	void PropagateForward();
	void PropagateBackward();
	void UpdateWeights();
};

#endif /* defined(__prediction__NeuralLayer__) */
