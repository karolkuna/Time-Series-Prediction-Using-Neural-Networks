//
//  RTRL.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__RTRL__
#define __prediction__RTRL__

#include <stdio.h>
#include <stdexcept>
#include "MemoryBlock.h"
#include "NeuralLayer.h"
#include "LearningAlgorithm.h"
#include "SimpleRecurrentNetwork.h"

using std::logic_error;

struct LayerPointer {
	NeuralLayer* layer;
	int offset;
	
	LayerPointer(NeuralLayer* layer, int offset) {
		this->layer = layer;
		this->offset = offset;
	}
};

class RTRL : public LearningAlgorithm {
private:
	SimpleRecurrentNetwork* m_network;
	float m_learningRate, m_momentumRate;
	
	int m_hiddenWeights;
	int m_outputWeights;
	int m_weights;
	
	int m_firstHiddenUnit, m_firstOutputUnit;
	int m_lastInputUnit, m_lastHiddenUnit, m_lastOutputUnit;
	int m_totalUnits;
	
	void UpdateWeights();
	
	LayerPointer GetWeightPointer(int from, int to) const;
	LayerPointer GetUnitPointer(int unit, int time) const;
	
	float Weight(int from, int to) const;
	float WeightDelta(int from, int to) const;
	void SetWeight(int from, int to, float value);
	void SetWeightDelta(int from, int to, float value);
	
	float WeightDerivative(int from, int to, int time, int unit) const;
	void SetWeightDerivative(int from, int to, int unit, int time, float value);
	
	float Activation(int unit, int time) const;
	void SetActivation(int unit, int time, float value);
	
	void CalculateDerivativesForWeight(int weightFrom, int weightTo);
	void UpdateWeight(int from, int to);
	
	bool IsRecurrentWeight(int from, int to) const;
	
public:
	
	RTRL(SimpleRecurrentNetwork* network, float learningRate, float momentumRate);
	~RTRL();
	
	MemoryBlock delta;
	MemoryBlock rtrlPastDerivatives;
	MemoryBlock rtrlDerivatives;
	MemoryBlock rtrlFutureDerivatives;
	MemoryBlock previousOutput;
	
	void Train(const MemoryBlock& target);
};

#endif /* defined(__prediction__RTRL__) */
