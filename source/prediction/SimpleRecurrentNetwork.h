//
//  RecurrentNetwork.h
//  prediction
//
//  Created by Karol Kuna on 28/03/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__SimpleRecurrentNetwork__
#define __prediction__SimpleRecurrentNetwork__

#include "NeuralNetwork.h"
#include "MemoryBlock.h"
#include "ActivationFunction.h"
#include "NeuralLayer.h"
#include "CWRecurrentNetwork.h"


class SimpleRecurrentNetwork : public CWRecurrentNetwork {
public:
	int units;
	int hiddenWeights;
	int outputWeights;
	int weights;
	
	NeuralLayer* contextLayer;
	NeuralLayer* hiddenLayer;
	
	MemoryBlock error;
	
	SimpleRecurrentNetwork(int inputUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction, LearningAlgorithm* learningAlgorithm);
};

#endif /* defined(__prediction__SimpleRecurrentNetwork__) */
