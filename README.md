# TSSUNet-MB
Identification of transcription start sites of bacterial sigma70 promoters by deep multi-task learning

TSSUNet-MB is a deep learning model developed to identify the TSSs of sigma70 promoters.

## Requirements

The following packages are required to rerun the provided code.
- pytorch
- tqdm
- torchsummary

## Usage

We provide two ipynb files which are separated into two different processes: training and evaluation. 
The weights of the TSSUNet-MB were not provided, to test and assess the model's performance, the user needs to re-train the model using the provided code.

- TSSUNet-MB-training.ipynb:
  - INPUT: Promoter sequences (located inside the data directory).
  - OUTPUT: A .pt file which store the TSSUNet-MB's weight.
  
- TSSUNet-MB-evaluates-plots.ipynb: Assess model's performance on the test set and generate relevant plots.
  - INPUT: Promoter sequences, model's weight file (.pt format).

## License

[MIT](https://choosealicense.com/licenses/mit/)
