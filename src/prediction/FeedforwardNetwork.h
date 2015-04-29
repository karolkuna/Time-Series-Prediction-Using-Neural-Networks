//
//  FeedforwardNetwork.h
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__FeedforwardNetwork__
#define __prediction__FeedforwardNetwork__

#include "MemoryBlock.h"
#include "ActivationFunctions.h"
#include "NeuralLayer.h"
#include "NeuralNetwork.h"


class FeedforwardNetwork : public NeuralNetwork {
public:
	ActivationFunction* activationFunction;
	std::vector<NeuralLayer*> layers;
	NeuralLayer* threshold;
	
	FeedforwardNetwork( int inputUnits, std::vector<int> hiddenLayerUnits, int outputUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate);
	~FeedforwardNetwork();
	
	void Propagate(MemoryBlock& input);
};

#endif /* defined(__prediction__FeedforwardNetwork__) */
