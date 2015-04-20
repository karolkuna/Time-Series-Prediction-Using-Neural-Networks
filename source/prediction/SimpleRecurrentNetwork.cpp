//
//  RecurrentNetwork.cpp
//  prediction
//
//  Created by Karol Kuna on 28/03/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "SimpleRecurrentNetwork.h"
vector<int> oneModule = {1};

SimpleRecurrentNetwork::SimpleRecurrentNetwork(int inputUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction, LearningAlgorithm* learningAlgorithm) : CWRecurrentNetwork(inputUnits, hiddenUnits, outputUnits, oneModule, activationFunction, NULL) {

	hiddenLayer = hiddenLayerModules[0];
	contextLayer = contextLayerModules[0];
	
	units = 1 + inputUnits + 2 * hiddenUnits + outputUnits;
	hiddenWeights = hiddenLayer->weights.size;
	outputWeights = outputLayer->weights.size;
	weights = hiddenWeights + outputWeights;
	
	error = MemoryBlock(outputUnits);
	
	this->learningAlgorithm = learningAlgorithm;
	learningAlgorithm->Init(this);
}
