//
//  FeedforwardNetwork.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__FeedforwardNetwork__
#define __prediction__FeedforwardNetwork__

#include <stdexcept>
#include "MemoryBlock.h"
#include "ActivationFunctions.h"
#include "NeuralLayer.h"
#include "NeuralNetwork.h"

using std::vector;
using std::logic_error;

class FeedforwardNetwork : public NeuralNetwork {
public:
	ActivationFunction* activationFunction;
	std::vector<NeuralLayer*> layers;
	NeuralLayer* threshold;
	
	FeedforwardNetwork( int inputUnits, const vector<int>& hiddenLayerUnits, int outputUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate);
	~FeedforwardNetwork();
	
	void Propagate(const MemoryBlock& input);
};

#endif /* defined(__prediction__FeedforwardNetwork__) */
