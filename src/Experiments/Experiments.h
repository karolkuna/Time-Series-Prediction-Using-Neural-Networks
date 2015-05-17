//
//  Experiments.cpp
//  Experiments
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__Experiments__
#define __prediction__Experiments__

#include <stdio.h>
#include <ctime>
#include <stdexcept>
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
const string DSTAT_DATASET_PATH = "Experiments/data/dstat.data";
const string MANIPULATOR_DATASET_PATH = "Experiments/data/dstat.data";


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

void ExperimentTDNNWindowSize(experiment_dataset dataset, int maxSize, vector<int> layers, float learningRate, float momentumRate);
void ExperimentTBPPTDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, float learningRate, float momentumRate);
void ExperimentRTRLUnits(experiment_dataset dataset, int maxHiddenUnits, float learningRate, float momentumRate);
void ExperimentCWDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, vector<int> clockRate, float learningRate, float momentumRate);

#endif