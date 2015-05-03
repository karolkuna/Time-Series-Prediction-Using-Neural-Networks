//
//  Backpropagation.h
//  prediction
//
//  Created by Karol Kuna on 29/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__Backpropagation__
#define __prediction__Backpropagation__

#include <stdio.h>
#include "MemoryBlock.h"
#include "NeuralLayer.h"
#include "FeedforwardNetwork.h"

class Backpropagation {
private:
	FeedforwardNetwork* m_network;
	
public:
	Backpropagation(FeedforwardNetwork* network);
	void Train(const MemoryBlock& target);

	float error;
};

#endif /* defined(__prediction__Backpropagation__) */
