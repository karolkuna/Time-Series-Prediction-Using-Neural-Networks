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
#include "ActivationFunctions.h"
#include "NeuralLayer.h"
#include "CWRecurrentNetwork.h"


class SimpleRecurrentNetwork : public CWRecurrentNetwork {
public:
	NeuralLayer* contextLayer;
	NeuralLayer* hiddenLayer;
	
	SimpleRecurrentNetwork(int inputUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction);
};

#endif /* defined(__prediction__SimpleRecurrentNetwork__) */
