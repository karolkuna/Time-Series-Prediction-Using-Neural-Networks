//
//  CWRecurrentNetwork.h
//  prediction
//
//  Created by Karol Kuna on 30/03/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__CWRecurrentNetwork__
#define __prediction__CWRecurrentNetwork__

#include <vector>
#include "ActivationFunctions.h"
#include "MemoryBlock.h"
#include "NeuralLayer.h"
#include "NeuralNetwork.h"

using std::vector;

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
