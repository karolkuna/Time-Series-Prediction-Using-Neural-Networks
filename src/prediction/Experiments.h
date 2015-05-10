//
//  Experiments.cpp
//  prediction
//
//  Created by Karol Kuna on 06/05/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__Experiments__
#define __prediction__Experiments__

#include <stdio.h>
#include <ctime>
#include "MemoryBlock.h"
#include "MemoryBlockView.h"
#include "DataSet.h"
#include "DstatParser.h"
#include "FeedforwardNetwork.h"
#include "SimpleRecurrentNetwork.h"
#include "CWRecurrentNetwork.h"
#include "Backpropagation.h"
#include "TBPTT.h"
#include "RTRL.h"

const int REPEAT_EXPERIMENTS = 10;
typedef enum experiment_dataset {
	GONIOMETRIC,
	NETWORK,
	MANIPULATOR
} experiment_dataset;

using std::cout;
using std::endl;
using std::vector;
using std::clock;
using std::clock_t;

void ExperimentLearningAndMomentumRate(NeuralNetwork* network, LearningAlgorithm* learning, float granularity, int min, int max, int steps);
void ExperimentTDNNWindowSize(experiment_dataset dataset, int maxSize, vector<int> layers, float learningRate, float momentumRate);
void ExperimentTBPPTDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, float learningRate, float momentumRate);
void ExperimentRTRLUnits(experiment_dataset dataset, int maxHiddenUnits, float learningRate, float momentumRate);
void ExperimentCWDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, vector<int> clockRate, float learningRate, float momentumRate);

#endif