# SDFVAE
SDFVAE is a robust and noisy-resilient anomaly detection method based on static and dynamic factorization, which is capable of explicitly learning the representations of time-invariant characteristics of multivariate KPIs, in addition to the time-varying characteristics.

## Getting Started
### Install dependencies (with python 3.5 or 3.6)

    pip install -r requirements.txt

### Data preprocessing
    python data_preprocess.py --raw_data_file data/machine-1-1-data.csv --label_file data/machine-1-1-label.csv --train_data_path data_processed/machine-1-1-train --test_data_path data_processed/machine-1-1-test --test_start_time 20190923005800
**Please refer to "data_preprocess/data_preprocessing_scripts.txt" for some details.**

### Training
    python trainer.py --dataset_path ../data_preprocess/data_processed/machine-1-1-train --data_nums  28253 --gpu_id 0 --log_path log_trainer/machine-1-1 --checkpoints_path model/machine-1-1 --n 38
**The detailed commands are given in "sdfvae/sdfvae_scripts.txt".**

### Testing
    nohup python tester.py --dataset_path ../data_preprocess/data_processed/machine-1-1-test --data_nums 28469 --gpu_id 0 --log_path log_tester/machine-1-1 --checkpoints_path model/machine-1-1 --n 38 --start_epoch 30 2>&1 &
**Refer to "sdfvae/sdfvae_scripts.txt" for details.**

### Evaluation
    nohup python evaluation.py --llh_path log_tester/machine-1-1 --log_path log_evaluator/machine-1-1 --n 38 --start_epoch 30 2>&1 &
**Please refer to "sdfvae/sdfvae_scripts.txt".**

## Training Losses
We give the example of SDFVAE training Losses on VoD1 dataset, the figure is in the directory named "training_losses". <br>
![image](https://github.com/dlagul/SDFVAE/blob/main/training_losses/vod1_training_losses.png) <br>
The results show that our model tends to converge around 30 epochs. <br>
Please refer to the directory named "log_trainer" for more details about training losses when you running SDFVAE.

## Testing results
We show an example of the log-likelihood tested on "machine-1-1", the picture is in the directory named "testing_results". <br>
![image](https://github.com/dlagul/SDFVAE/blob/main/testing_results/sdfvae-anomaly-score-on-machine-1-1.jpg) <br>
Regions highlighted in red represent the ground-truth anomaly segments. The blue line is the log-likelihood or anomaly score output by SDFVAE. <br>
We are able to determine the anomalies via a specific threshold, due to the lower the log-likelihood the higher the anomaly score. <br> 
Since we do not focus on the thresholding technique, in practice, the threshold for detecting anomaly from the anomaly score can be selected by the best F1-score. <br>
The detailed testing results can be found in the directory named "log_tester".

## Evaluation results
The detailed resuts of evaluation are in the directory named "log_evaluator". <br>
We obtain all F1-score by enumerating all thresholds and use the best F1-score as the final score.


# Exp_datasets

## Dataset Information

### Data format
There are 2 CSV files of each dataset, one is the KPIs data file, the other is the corresponding ground-truth file. <br>
The KPIs data file has the following format: <br>
* The first row is the timestamp <br>
* The other rows is the KPIs values, and each row corresponds to a KPI <br>

Timestamp | 20181001000500  | 20181001001000 | 20181001001500 | ...
--- | --- | --- | --- | ---
KPI_1 | 0.4423121844530866 | 0.46444977152338346 | 0.4186436700242946 | ...
... | ... | ... | ... | ... 
KPI_n | 0.37977501905111494 | 0.39892597116922146 | 0.36615750635812155 | ...

The ground-truth file has the following format: <br>
* The first row is the timestamp <br>
* The second row is the label, 0 for normal and 1 for abnormal <br>

Timestamp | 20181001000500  | 20181001001000 | 20181001001500 | ...
--- | --- | --- | --- | ---
label | 0 | 0 | 1 | ...

### Public Dataset 
The public dataset (SMD) used in our evaluation experiments as well as its detailed description can be found in web sites:
https://github.com/NetManAIOps/OmniAnomaly

In order to make it easy for reviewers to run our code on SMD, we select 2 of 28 namely "machine-1-1.txt" and "machine-1-5.txt" as an example. We add the timestamp to the two datasets to create the format required by our data preprocessing code. Please do not forget to add the timestamp if you want to test SDFVAE on others datasets. *It should be noted that SDFVAE never utilise any information of these timestamps to improve its performance.* 
