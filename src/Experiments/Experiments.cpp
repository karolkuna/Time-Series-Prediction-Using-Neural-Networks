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

#include "Experiments.h"

void ExperimentTDNNWindowSize(experiment_dataset dataset, int maxSize, vector<int> layers, float learningRate, float momentumRate) {

	cout << "TDNN Window Size Experiment" << endl;
	cout << "Window Size | Total Error | Time" << endl;
	
	LogisticFunction logistic;
	
	for (int slidingWindowSize = 1; slidingWindowSize <= maxSize; slidingWindowSize++) {
		float averageTotalError = 0;
		float averageTimeDuration = 0;
		
		for (int i = 0; i < REPEAT_EXPERIMENTS; i++) {
			int historySize;
			int targetSize;
			int steps;
			
			DataSet manipulator(MANIPULATOR_DATASET_PATH, true);
			DstatParser dstat(DSTAT_DATASET_PATH);
			
			if (dataset == GONIOMETRIC) {
				historySize = slidingWindowSize * 1;
				targetSize = 1;
				steps = 1000;
			} else if (dataset == NETWORK) {
				historySize = slidingWindowSize * dstat.output.size;
				targetSize = 2;
				steps = 3600;
			} else if (dataset == MANIPULATOR) {
				historySize = slidingWindowSize * (manipulator.input.size + manipulator.output.size);
				targetSize = 3;
				steps = 18000;
			}
			
			MemoryBlock history(historySize);
			MemoryBlock target(targetSize);
			
			target.Fill(0);
			history.Fill(0);
			
			FeedforwardNetwork network = FeedforwardNetwork(historySize, layers, targetSize, &logistic, learningRate, momentumRate);
			Backpropagation bp(&network);
			
			float totalError = 0;
			
			clock_t start = clock();
			
			for (int t = 0; t < steps; t++) {
				if (dataset == GONIOMETRIC) {
					history.RightShift(1);
					target.CopyTo(history, 0, 0, 1);
					
					target.data[0] = 0.5f * (1 + sinf(t) * cosf(2*t));
				} else if (dataset == NETWORK) {
					history.RightShift(dstat.output.size);
					dstat.output.CopyTo(history, 0, 0, dstat.output.size);
					
					dstat.Parse();
					dstat.Preprocess();
					dstat.output.CopyTo(target, dstat.recvIndex, 0, 2);
				} else if (dataset == MANIPULATOR) {
					history.RightShift(manipulator.input.size + manipulator.output.size);
					manipulator.input.Add(1);
					manipulator.input.Multiply(0.5f);
					manipulator.input.CopyTo(history, 0, 0, manipulator.input.size);
					manipulator.output.CopyTo(history, 0, manipulator.input.size, manipulator.output.size);
					
					manipulator.Read();
					manipulator.output.CopyTo(target, 3, 0, 3);
				}
				
				
				network.Propagate(history);
				bp.Train(target);
				
				totalError += bp.error;
			}
			
			averageTotalError += totalError;
			averageTimeDuration += (float) (std::clock() - start) / (float)(CLOCKS_PER_SEC);
		}
		
		averageTotalError /= (float) REPEAT_EXPERIMENTS;
		averageTotalError /= 2.0f;
		averageTimeDuration /= (float) REPEAT_EXPERIMENTS;
		
		cout << slidingWindowSize << " " << averageTotalError << " " << averageTimeDuration << endl;
	}
	
	cout << endl;
}

void ExperimentTBPPTDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, float learningRate, float momentumRate) {
	
	cout << "TBPTT Unfolding Depth Experiment" << endl;
	cout << "Unfolding Depth | Total Error | Time" << endl;
	
	LogisticFunction logistic;
	
	for (int depth = 1; depth <= maxDepth; depth++) {
		float averageTotalError = 0;
		float averageTimeDuration = 0;
		
		for (int i = 0; i < REPEAT_EXPERIMENTS; i++) {
			int inputSize;
			int targetSize;
			int steps;
			
			DataSet manipulator(MANIPULATOR_DATASET_PATH, true);
			DstatParser dstat(DSTAT_DATASET_PATH);
			
			if (dataset == GONIOMETRIC) {
				inputSize = 1;
				targetSize = 1;
				steps = 1000;
			} else if (dataset == NETWORK) {
				inputSize = dstat.output.size;
				targetSize = 2;
				steps = 3600;
			} else if (dataset == MANIPULATOR) {
				inputSize = manipulator.input.size + manipulator.output.size;
				targetSize = 3;
				steps = 18000;
			}
			
			MemoryBlock input(inputSize);
			MemoryBlock target(targetSize);
			
			input.Fill(0);
			target.Fill(0);
			
			SimpleRecurrentNetwork network = SimpleRecurrentNetwork(inputSize, hiddenUnits, targetSize, &logistic);
			TBPTT tbptt = TBPTT(&network, learningRate, momentumRate, depth);
			
			float totalError = 0;
			
			clock_t start = clock();
			
			for (int t = 0; t < steps; t++) {
				if (dataset == GONIOMETRIC) {
					target.CopyTo(input);
					target.data[0] = 0.5f * (1 + sinf(t) * cosf(2*t));
				} else if (dataset == NETWORK) {
					dstat.output.CopyTo(input, 0, 0, dstat.output.size);
					
					dstat.Parse();
					dstat.Preprocess();
					dstat.output.CopyTo(target, dstat.recvIndex, 0, 2);
				} else if (dataset == MANIPULATOR) {
					manipulator.input.Add(1);
					manipulator.input.Multiply(0.5f);
					manipulator.input.CopyTo(input, 0, 0, manipulator.input.size);
					manipulator.output.CopyTo(input, 0, manipulator.input.size, manipulator.output.size);
					
					manipulator.Read();
					manipulator.output.CopyTo(target, 3, 0, 3);
				}
				
				
				network.Propagate(input);
				tbptt.Train(target);
				
				totalError += tbptt.error;
			}
			
			averageTotalError += totalError;
			averageTimeDuration += (float) (std::clock() - start) / (float)(CLOCKS_PER_SEC);
		}
		
		averageTotalError /= (float) REPEAT_EXPERIMENTS;
		averageTotalError /= 2.0f;
		averageTimeDuration /= (float) REPEAT_EXPERIMENTS;
		
		cout << depth << " " << averageTotalError << " " << averageTimeDuration << endl;
	}
	
	cout << endl;
}

void ExperimentRTRLUnits(experiment_dataset dataset, int hiddenUnits, float learningRate, float momentumRate) {
	
	cout << "RTRL Number of Hidden Units Experiment" << endl;
	cout << "Number of Hidden Units | Total Error | Time" << endl;
	
	LogisticFunction logistic;
	float averageTotalError = 0;
	float averageTimeDuration = 0;
		
	for (int i = 0; i < REPEAT_EXPERIMENTS; i++) {
		int inputSize;
		int targetSize;
		int steps;
			
		DataSet manipulator(MANIPULATOR_DATASET_PATH, true);
		DstatParser dstat(DSTAT_DATASET_PATH);
			
		if (dataset == GONIOMETRIC) {
			inputSize = 1;
			targetSize = 1;
			steps = 1000;
		} else if (dataset == NETWORK) {
			inputSize = dstat.output.size;
			targetSize = 2;
			steps = 3600;
		} else if (dataset == MANIPULATOR) {
			inputSize = manipulator.input.size + manipulator.output.size;
			targetSize = 3;
			steps = 18000;
		}
		
		MemoryBlock input(inputSize);
		MemoryBlock target(targetSize);
			
		input.Fill(0);
		target.Fill(0);
			
		SimpleRecurrentNetwork network = SimpleRecurrentNetwork(inputSize, hiddenUnits, targetSize, &logistic);
		RTRL rtrl(&network, learningRate, momentumRate);
			
		float totalError = 0;
			
		clock_t start = clock();
			
		for (int t = 0; t < steps; t++) {
			if (dataset == GONIOMETRIC) {
				target.CopyTo(input);
				target.data[0] = 0.5f * (1 + sinf(t) * cosf(2*t));
			} else if (dataset == NETWORK) {
				dstat.output.CopyTo(input, 0, 0, dstat.output.size);
				
				dstat.Parse();
				dstat.Preprocess();
				dstat.output.CopyTo(target, dstat.recvIndex, 0, 2);
			} else if (dataset == MANIPULATOR) {
				manipulator.input.Add(1);
				manipulator.input.Multiply(0.5f);
				manipulator.input.CopyTo(input, 0, 0, manipulator.input.size);
				manipulator.output.CopyTo(input, 0, manipulator.input.size, manipulator.output.size);
					
				manipulator.Read();
				manipulator.output.CopyTo(target, 3, 0, 3);
			}
				
			
			network.Propagate(input);
			rtrl.Train(target);
				
			totalError += rtrl.error;
		}
			
		averageTotalError += totalError;
		averageTimeDuration += (float) (std::clock() - start) / (float)(CLOCKS_PER_SEC);
	}
		
	averageTotalError /= (float) REPEAT_EXPERIMENTS;
	averageTotalError /= 2.0f;
	averageTimeDuration /= (float) REPEAT_EXPERIMENTS;
		
	cout << hiddenUnits << " " << averageTotalError << " " << averageTimeDuration << endl;
}

void ExperimentCWDepth(experiment_dataset dataset, int maxDepth, int hiddenUnits, vector<int> clockRate, float learningRate, float momentumRate) {
	LogisticFunction logistic;
	
	cout << "CW-RNN trained by TBPTT Unfolding Depth Experiment" << endl;
	cout << "Unfolding Depth | Total Error | Time" << endl;
	
	for (int depth = 1; depth <= maxDepth; depth++) {
		float averageTotalError = 0;
		float averageTimeDuration = 0;
		
		for (int i = 0; i < REPEAT_EXPERIMENTS; i++) {
			int inputSize;
			int targetSize;
			int steps;
			
			DataSet manipulator(MANIPULATOR_DATASET_PATH, true);
			DstatParser dstat(DSTAT_DATASET_PATH);
			
			if (dataset == GONIOMETRIC) {
				inputSize = 1;
				targetSize = 1;
				steps = 1000;
			} else if (dataset == NETWORK) {
				inputSize = dstat.output.size;
				targetSize = 2;
				steps = 3600;
			} else if (dataset == MANIPULATOR) {
				inputSize = manipulator.input.size + manipulator.output.size;
				targetSize = 3;
				steps = 18000;
			}
			
			MemoryBlock input(inputSize);
			MemoryBlock target(targetSize);
			
			input.Fill(0);
			target.Fill(0);
			
			CWRecurrentNetwork network = CWRecurrentNetwork(inputSize, hiddenUnits, targetSize, clockRate, &logistic);
			TBPTT tbptt = TBPTT(&network, learningRate, momentumRate, depth);
			
			float totalError = 0;
			
			clock_t start = clock();
			
			for (int t = 0; t < steps; t++) {
				if (dataset == GONIOMETRIC) {
					target.CopyTo(input);
					target.data[0] = 0.5f * (1 + sinf(t) * cosf(2*t));
				} else if (dataset == NETWORK) {
					dstat.output.CopyTo(input, 0, 0, dstat.output.size);
					
					dstat.Parse();
					dstat.Preprocess();
					dstat.output.CopyTo(target, dstat.recvIndex, 0, 2);
				} else if (dataset == MANIPULATOR) {
					manipulator.input.Add(1);
					manipulator.input.Multiply(0.5f);
					manipulator.input.CopyTo(input, 0, 0, manipulator.input.size);
					manipulator.output.CopyTo(input, 0, manipulator.input.size, manipulator.output.size);
					
					manipulator.Read();
					manipulator.output.CopyTo(target, 3, 0, 3);
				}
				
				
				network.Propagate(input);
				tbptt.Train(target);
				
				totalError += tbptt.error;
			}
			
			averageTotalError += totalError;
			averageTimeDuration += (float) (std::clock() - start) / (float)(CLOCKS_PER_SEC);
		}
		
		averageTotalError /= (float) REPEAT_EXPERIMENTS;
		averageTotalError /= 2.0f;
		averageTimeDuration /= (float) REPEAT_EXPERIMENTS;
		
		cout << depth << " " << averageTotalError << " " << averageTimeDuration << endl;
	}
	cout << endl;
}
