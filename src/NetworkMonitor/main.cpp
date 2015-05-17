//
//  main.cpp
//  NetworkMonitor
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <fstream>
#include "FeedforwardNetwork.h"
#include "CWRecurrentNetwork.h"
#include "SimpleRecurrentNetwork.h"
#include "TBPTT.h"
#include "RTRL.h"
#include "MemoryBlockView.h"
#include "DataSet.h"
#include "DstatParser.h"


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
	//runAllTests();
	//runAllExperiments();
	
	int LEARNING_PERIOD = 10;

	cout << "NetworkMonitor started..." << endl;
	
	std::ofstream outfile;
	outfile.open("anomalies.log", std::ios_base::app);
	outfile << "NetworkMonitor started..." << endl;
	outfile.flush();
	
	DstatParser dstat;
	
	MemoryBlock input(dstat.output.size);
	MemoryBlock target(2);
	
	input.Fill(0);
	target.Fill(0);
	
	LogisticFunction logistic;
	SimpleRecurrentNetwork srn(input.size, 128, target.size, &logistic);
	TBPTT tbptt(&srn, 0.01, 0.9, 5);
	
	int step = 0;
	double totalError = 0;
	
	while (dstat.Parse()) {
		step++;
		
		dstat.Preprocess();
		
		dstat.output.CopyTo(input, 0, 0, dstat.output.size);
		dstat.output.CopyTo(target, dstat.recvIndex, 0, 2);
		
		tbptt.Train(target);
		srn.Propagate(input);
		
		
		if (step > LEARNING_PERIOD) {
			cout << "Current error: " << tbptt.error << endl;
			totalError += tbptt.error;
			
			float average = (float) (totalError / (double) (step - LEARNING_PERIOD));
			if (tbptt.error > 2 * average) {
				cout << "Anomaly detected!" << endl;
				outfile << "Anomaly detected! Step #";
				outfile << step << ": " << dstat.line << endl;
				
				outfile.flush();
			}
		}
	}
	
	
    return 0;
}
