# Complete Cycle-of-Learning (CoL)

<center><img src="docs/Screenshot.png" alt="CoL AirSim screenshot." width="400"/></center>

Cycle-of-Learning applied to teaching a quadrotor how to land on top of a landing pad based on human interaction and reinforcement learning.

## Installation

### Setup dependencies

All dependencies will be contained in a virtual environment. A Docker option will be available in future.

Install _pip_ and _virtualenv_ to handle all Python3 dependencies:  
```sudo apt-get install python3-pip```  
```python3 -m pip install --user virtualenv```  

Create a new virtual environment:  
```python3 -m venv ~/venvs/CoL```

Activate the new environment and install dependencies:  
```source ~/venvs/CoL/bin/activate```  
```pip install wheel```  
```pip install -r requirements.txt```

### Setup AirSim

Download AirSim binary file from: ```https://drive.google.com/file/d/1GO7eb2JzSmnsrW62_J7X14i-ipfImxqp/view?usp=sharing```.  

Extract it and copy the address of the ```HRI_AirSim.sh``` file to line 5 of ```run.sh``` script.  

Copy the ```settings.json``` file to ```~/Documents/AirSim/settings.json```.   

Test the binary file by running ```./HRI_AirSim.sh``` from its folder. The AirSim environment should start with the quadrotor landed on top of the landing pad.

### Setup human joystick

The human can provide additional demonstrations and interventions by controlling the vehicle using a Xbox One controller. Make sure it is plugged in before starting the AirSim environment. It should also work with a Xbox 360 joystick but this has not been tested yet.

## Training

After setting up all dependencies and AirSim file, activate the virtual environment ```source ~/venvs/CoL/bin/activate``` and run ```./run.sh``` to automatically start the AirSim environment and training script. The default script loads previous 20 human trajectories, pretrain the actor and critic, and update these models using reinforcement learning. At any time, the human can intervene by pressing the LB button in the Xbox One controller and controlling the vehicle using the left and right sticks.  

Default training hyperparameters can be changed inside ```./run.sh```.

## Citation

The Cycle-of-Learning concept:  
```
@article{waytowich2018cycle,
  author={Nicholas R. Waytowich and Vinicius G. Goecks and Vernon J. Lawhern},
  title={Cycle-of-Learning for Autonomous Systems from Human Interaction},
  journal={CoRR},
  volume={abs/1808.09572},
  year={2018},
  url={http://arxiv.org/abs/1808.09572},
  archivePrefix={arXiv},
  eprint={1808.09572}
}
```

Combining in real-time demonstrations and interventions for improve task performance using fewer samples:  
```
@article{goecks2018efficiently,
  author={Vinicius G. Goecks and Gregory M. Gremillion and Vernon J. Lawhern and John Valasek and Nicholas R. Waytowich},
  title={Efficiently Combining Human Demonstrations and Interventions for Safe Training of Autonomous Systems in Real-Time},
  journal={CoRR},
  volume={abs/1810.11545},
  year={2018},
  url={http://arxiv.org/abs/1810.11545},
  archivePrefix={arXiv},
  eprint={1810.11545}
}
```

Transitioning from policies learned through demonstrations and interventions to reinforcement learning:  
```
@misc{goecks2019integrating,
    title={Integrating Behavior Cloning and Reinforcement Learning for Improved Performance in Sparse Reward Environments},
    author={Vinicius G. Goecks and Gregory M. Gremillion and Vernon J. Lawhern and John Valasek and Nicholas R. Waytowich},
    year={2019},
    eprint={1910.04281},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
