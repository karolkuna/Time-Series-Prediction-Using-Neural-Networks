//
//  main.cpp
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "FeedforwardNetwork.h"
#include "CWRecurrentNetwork.h"
#include "SimpleRecurrentNetwork.h"
#include "TBPTT.h"
#include "RTRL.h"
#include "MemoryBlockView.h"
#include "DataSet.h"
#include "DstatParser.h"
#include "Experiments.h"


using std::cout;
using std::endl;

void activationFunctionTests() {
	ActivationFunction* logistic = new LogisticFunction();
	LogisticFunction logistic2;
	ActivationFunction& logistic3 = *logistic;
	
	assert((*logistic)(0.5) != 0);
	assert((*logistic)(0.5) == logistic2(0.5));
	assert((*logistic)(0.5) == logistic3(0.5));
}

void memoryBlockTests() {
	MemoryBlock a = MemoryBlock(5); assert(a.size == 5);
	a.Fill(1); assert(a.Sum() == 5);
	a.LeftShift(0); assert(a.Sum() == 5);
	a.RightShift(0); assert(a.Sum() == 5);
	a.RightShift(2); assert(a.data[0] == 0 && a.data[1] == 0 && a.data[2] == 1 && a.data[3] == 1 & a.data[4] == 1);
	a.LeftShift(2); assert(a.data[0] == 1 && a.data[1] == 1 && a.data[2] == 1 && a.data[3] == 0 & a.data[4] == 0);
}

void runAllTests() {
	activationFunctionTests();
	memoryBlockTests();
	std::cout << "All tests passed\n";
}
 
int main(int argc, const char * argv[]) {	
	runAllTests();
	
	ExperimentTDNNWindowSize(GONIOMETRIC, 20, {128}, 0.05, 0.9);
	ExperimentTDNNWindowSize(GONIOMETRIC, 20, {256}, 0.05, 0.9);
	ExperimentTDNNWindowSize(GONIOMETRIC, 20, {512}, 0.05, 0.9);
	
	ExperimentTDNNWindowSize(NETWORK, 30, {128}, 0.001, 0.9);
	ExperimentTDNNWindowSize(NETWORK, 30, {256}, 0.001, 0.9);
	ExperimentTDNNWindowSize(NETWORK, 30, {512}, 0.001, 0.9);
	
	ExperimentTDNNWindowSize(MANIPULATOR, 20, {128}, 0.01, 0.9);
	ExperimentTDNNWindowSize(MANIPULATOR, 20, {256}, 0.01, 0.9);
	ExperimentTDNNWindowSize(MANIPULATOR, 20, {512}, 0.01, 0.9);

	ExperimentTBPPTDepth(GONIOMETRIC, 20, 64, 0.01, 0.9);
	ExperimentTBPPTDepth(GONIOMETRIC, 20, 128, 0.01, 0.9);
	ExperimentTBPPTDepth(GONIOMETRIC, 20, 256, 0.01, 0.9);

	ExperimentTBPPTDepth(NETWORK, 20, 32, 0.001, 0.9);
	ExperimentTBPPTDepth(NETWORK, 20, 64, 0.001, 0.9);
	ExperimentTBPPTDepth(NETWORK, 20, 128, 0.001, 0.9);
	
	ExperimentTBPPTDepth(MANIPULATOR, 20, 32, 0.01, 0.9);
	ExperimentTBPPTDepth(MANIPULATOR, 20, 64, 0.01, 0.9);
	ExperimentTBPPTDepth(MANIPULATOR, 20, 128, 0.01, 0.9);
	
	ExperimentRTRLUnits(GONIOMETRIC, 8, 0.1, 0.9);
	ExperimentRTRLUnits(GONIOMETRIC, 16, 0.1, 0.9);
	ExperimentRTRLUnits(GONIOMETRIC, 32, 0.1, 0.9);
	
	ExperimentRTRLUnits(NETWORK, 8, 0.1, 0.9);
	ExperimentRTRLUnits(NETWORK, 16, 0.1, 0.9);
	ExperimentRTRLUnits(NETWORK, 32, 0.1, 0.9);
	
	ExperimentRTRLUnits(MANIPULATOR, 8, 0.1, 0.9);
	ExperimentRTRLUnits(MANIPULATOR, 16, 0.1, 0.9);
	ExperimentRTRLUnits(MANIPULATOR, 32, 0.1, 0.9);
	
	ExperimentCWDepth(GONIOMETRIC, 20, 16, {1,2,4,8}, 0.01, 0.9);
	ExperimentCWDepth(GONIOMETRIC, 20, 32, {1,2,4,8}, 0.01, 0.9);
	ExperimentCWDepth(GONIOMETRIC, 20, 64, {1,2,4,8}, 0.01, 0.9);

	ExperimentCWDepth(NETWORK, 20, 8, {1,2,4,8}, 0.001, 0.9);
	ExperimentCWDepth(NETWORK, 20, 16, {1,2,4,8}, 0.001, 0.9);
	ExperimentCWDepth(NETWORK, 20, 32, {1,2,4,8}, 0.001, 0.9);
	
	ExperimentCWDepth(MANIPULATOR, 20, 8, {1,2,4,8}, 0.01, 0.9);
	ExperimentCWDepth(MANIPULATOR, 20, 16, {1,2,4,8}, 0.01, 0.9);
	ExperimentCWDepth(MANIPULATOR, 20, 32, {1,2,4,8}, 0.01, 0.9);

    return 0;
}
