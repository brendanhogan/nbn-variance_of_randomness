# Variance of Randomness in Neural Network Training

This repository contains the code for the experiments discussed in the Neuron by Neuron blog article: "How much variance does randomness (initializations & data shuffling) introduce?"

[Blog Post Link](https://neuron-by-neuron.ghost.io/variance_of_randomness/)

## Overview

This project investigates the impact of randomness in neural network training, specifically focusing on initialization and data shuffling. We conduct three experiments using ResNet-18 on the CIFAR-10 dataset, each designed to isolate different aspects of randomness in the training process. These experiments were conducted using 100 hours of compute time on an NVIDIA A100 GPU, rented from Lambda Labs.

## Experiments

### Experiment 1: Standard Training (exp1.py)

This experiment runs 1000 training sessions with normal settings to observe overall performance variance. It uses random initialization and random data shuffling for each run.

Key features:
- Random network initialization for each run
- Random data shuffling for each run
- 1000 complete training sessions

### Experiment 2: Fixed Data Order (exp2.py)

This experiment isolates the effect of network initialization by keeping the data order constant across all runs.

Key features:
- Random network initialization for each run
- Fixed data order across all runs
- 1000 complete training sessions

### Experiment 3: Fixed Network Initialization (exp3.py)

This experiment isolates the effect of data order by keeping the network initialization constant across all runs.

Key features:
- Fixed network initialization across all runs
- Random data shuffling for each run
- 1000 complete training sessions

## Usage

Each experiment script (exp1.py, exp2.py, exp3.py) can be run with the following flags:

- `--run_experiment`: Run the training process
- `--plot`: Generate plots from the results
- `--output_dir`: Specify the directory to save output (default is 'exp1output', 'exp2output', or 'exp3output')
- `--number_of_runs`: Specify the number of runs for the experiment (default is 1000)

Example usage:

```bash
python exp1.py --run_experiment --plot --output_dir exp1output --number_of_runs 1000
```

```bash
python exp2.py --run_experiment --plot --output_dir exp2output --number_of_runs 1000
```

```bash
python exp3.py --run_experiment --plot --output_dir exp3output --number_of_runs 1000
```

## Results and Analysis

After running the experiments, the results are analyzed and visualized using various plots. These plots help in understanding the impact of different factors on the training process and model performance.

### Generated Plots

For each experiment, the following plots are generated:

1. **Training Loss Curves**: Shows the progression of training loss across epochs for all runs.
2. **Training Accuracy Curves**: Displays the improvement in training accuracy over epochs for all runs.
3. **Test Accuracy Distribution**: Illustrates the distribution of final test accuracies across all runs.
4. **Test Loss Distribution**: Presents the distribution of final test losses for all runs.

### Interpreting the Results

The comparison of these plots across the three experiments provides insights into:

1. **Overall Variability**: Experiment 1 shows the combined effect of both network initialization and data order randomization.
2. **Impact of Network Initialization**: Experiment 2 isolates the effect of network initialization by keeping data order constant.
3. **Influence of Data Order**: Experiment 3 demonstrates the impact of data order by maintaining a fixed network initialization.

By analyzing these results, we can draw conclusions about:

- The relative importance of network initialization versus data order in training variability.
- The consistency of model performance across different runs.
- Potential strategies for improving model robustness and reducing training variability.

## Conclusion

This project provides a comprehensive analysis of the factors influencing the variability in neural network training. The insights gained from these experiments can be valuable for:

- Developing more robust training strategies.
- Understanding the limitations and strengths of current training practices.
- Guiding future research in improving the consistency and reliability of neural network models.

For detailed results and analysis, please refer to the individual experiment outputs and generated plots in their respective output directories.

## Future Work

Potential areas for future investigation include:

1. Extending the experiments to diverse architectures (e.g., transformers, GANs) and modalities (e.g., vision, language, audio).
2. Investigating variability specifically in early stages of training versus late stages, to understand critical periods of model development.
3. Analyzing gradient flows throughout the training process to identify patterns or anomalies that contribute to variability.
4. Exploring the relationship between model complexity and training variability across different stages of learning.
5. Developing techniques to stabilize training trajectories without sacrificing model performance or generalization ability.

































#