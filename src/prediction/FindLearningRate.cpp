//
//  FindLearningRate.cpp
//  prediction
//
//  Created by Karol Kuna on 30/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include <stdio.h>
#include "MemoryBlock.h"
#include "NeuralNetwork.h"
#include "LearningAlgorithm.h"

using std::cout;
using std::endl;

void FindLearningAndMomentumRate(NeuralNetwork* network, LearningAlgorithm* learning, float granularity, int min, int max, int steps) {
	MemoryBlock input(1);
	MemoryBlock target(1);
	
	for (float learningRate = min; learningRate <= max; learningRate += granularity) {
		for (float momentumRate = 0; momentumRate <= 1; momentumRate += granularity) {
			float totalError = 0;
			for (int step = 0; step < steps; step++) {
				network->Propagate(input);
				learning->Train(target);
				
				totalError += learning->error;
			}
			
			cout << totalError << " ";
		}
		cout << endl;
	}
}
