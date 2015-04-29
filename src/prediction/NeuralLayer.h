//
//  NeuralLayer.h
//  prediction
//
//  Created by Karol Kuna on 22/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__NeuralLayer__
#define __prediction__NeuralLayer__

#include <vector>
#include "ActivationFunctions.h"
#include "MemoryBlock.h"

using std::vector;

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
	
	void ConnectTo(NeuralLayer* nextLayer);
	
	void SetActivation(MemoryBlock& input);
	void GetActivation(MemoryBlock& output);
	void SetTarget(MemoryBlock& target);
	
	void PropagateForward();
	void PropagateBackward();
	void UpdateWeights();
};

#endif /* defined(__prediction__NeuralLayer__) */
