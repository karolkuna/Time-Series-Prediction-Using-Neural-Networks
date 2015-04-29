//
//  RTRL.h
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__RTRL__
#define __prediction__RTRL__

#include <stdio.h>
#include "MemoryBlock.h"
#include "NeuralLayer.h"
#include "LearningAlgorithm.h"
#include "SimpleRecurrentNetwork.h"

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
	
	LayerPointer GetWeightPointer(int from, int to);
	LayerPointer GetUnitPointer(int unit, int time);
	
	float Weight(int from, int to);
	float WeightDelta(int from, int to);
	void SetWeight(int from, int to, float value);
	void SetWeightDelta(int from, int to, float value);
	
	float WeightDerivative(int from, int to, int time, int unit);
	void SetWeightDerivative(int from, int to, int unit, int time, float value);
	
	float Activation(int unit, int time);
	void SetActivation(int unit, int time, float value);
	
	void CalculateDerivativesForWeight(int weightFrom, int weightTo);
	void UpdateWeight(int from, int to);
	
	bool IsRecurrentWeight(int from, int to);
	
public:
	
	RTRL(SimpleRecurrentNetwork* network, float learningRate, float momentumRate);
	~RTRL();
	
	MemoryBlock delta;
	MemoryBlock rtrlPastDerivatives;
	MemoryBlock rtrlDerivatives;
	MemoryBlock rtrlFutureDerivatives;
	MemoryBlock previousOutput;
	
	void Train(MemoryBlock& target);
};

#endif /* defined(__prediction__RTRL__) */
