//
//  README.txt
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

Experiments
- Includes all experiments tested in the thesis and their datasets
- Instructions:
	UNIX (tested on Debian 7.0 x64):
		1. cp NeuralPredictionLibrary/* Experiments
		2. g++ Experiments/*.cpp -std=c++11 -o experiments
		3. ./experiments

	Mac:
		1. Open Experiments.xcodeproj
		2. Update DSTAT_DATASET_PATH and MANIPULATOR_DATASET_PATH constants in Experiments.h to your absolute(!) paths of dataset files 'Experiments/data/dstat.data' and 'Experiments/data/manipulator.data' respectively
		3. Run it


Network Monitor
- Works with assistance of Dstat resource monitoring tool by piping output of Dstat to input of Network Monitor
- Network Monitor automatically parses and preprocesses Dstat output format
- Simple recurrent network trained online by truncated backpropagation through time is tasked to predict volume of incoming and outgoing traffic in next time step
- After LEARNING_PERIOD (configurable in main.cpp, default 10) steps starts printing current prediction error to console
- If current prediction error exceeds 2x the average, this unexpected event is logged in anomalies.log file
- Instructions:
	UNIX (tested on Debian 7.0 x64):
		1. Install Dstat (Debian: sudo apt-get install dstat)
		2. cp NeuralPredictionLibrary/* NetworkMonitor
		3. g++ NetworkMonitor/*.cpp -std=c++11 -o networkmonitor
		4. dstat | ./networkmonitor

Robot Arm Demo
- Works together with BEPUphysics Robot Arm Demo (RobotArmDemo/BEPUphysicsDemos/BEPUphysicsDemos/Demos/RobotArmDemo.cs), other demos are disabled
- From the NeuralPredictionLibrary written in unmanaged C++, managed C++/CLR ManagedNeuralPredictionLibrary DLL is created and referenced in Robot Arm Demo
- Simple recurrent network trained online by truncated backpropagation through time is tasked to predict next position of robot arm's claw
- Line is drawn in the world between current and predicted position of the claw
- Instructions:
	Windows
		1. Download and install MXA Game Studio which is necessary to build and launch BEPUPhysics from https://mxa.codeplex.com/releases
		2. Open ManagedNeuralPredictionLibrary/ManagedNeuralPredictionLibrary.sln solution in Visual Studio and build it in Release mode. This will create a DLL file that's needed later
		3. Open RobotArmDemo/BEPUphysicsDemos.sln solution and run it
