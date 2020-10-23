# SDFVAE
SDFVAE is a robust and noisy-resilient anomaly detection method based on static and dynamic factorization, which is capable of explicitly learning the representations of time-invariant characteristics of multivariate KPIs, in addition to the time-varying characteristics.

## Getting Started
### Install dependencies (with python 3.5 or 3.6)

    pip install -r requirements.txt

### Data preprocessing
    python data_preprocess.py --raw_data_file data/machine-1-1-data.csv --label_file data/machine-1-1-label.csv --train_data_path data_processed/machine-1-1-train --test_data_path data_processed/machine-1-1-test --test_start_time 20190923005800
**Please refer to "scripts.txt" for some details.**

### Training
    python trainer.py --dataset_path ../data_preprocess/data_processed/machine-1-1-train --data_nums  28253 --gpu_id 0 --log_path log_trainer/machine-1-1 --checkpoints_path model/machine-1-1 --n 38
**The detailed commands are given in "scripts.txt".**

### Testing
    nohup python tester.py --dataset_path ../data_preprocess/data_processed/machine-1-1-test --data_nums 28469 --gpu_id 0 --log_path log_tester/machine-1-1 --checkpoints_path model/machine-1-1 --n 38 --start_epoch 30 2>&1 &
**Refer to "scripts.txt" for details.**

### Evaluation
    nohup python evaluation.py --llh_path log_tester/machine-1-1 --log_path log_evaluator/machine-1-1 --n 38 --start_epoch 30 2>&1 &
**Please refer to "scripts.txt".**

## Training Losses
We give the example of SDFVAE training Losses on VoD1 dataset, the figure is in the directory named "training_losses". <br>
The results show that our model tends to converge within 30 epochs. <br>
Please refer to the directory named "log_trainer" for more details about training losses when you running SDFVAE.

## Testing results
We show an example of the log-likelihood tested on "machine-1-1", the picture is in the directory named "testing_results". <br>
Regions highlighted in red represent the ground-truth anomaly segments. The blue line is the log-likelihood or anomaly score output by SDFVAE. <br>
We are able to determine the anomalies via a specific threshold, due to the lower the log-likelihood the higher the anomaly score. <br> 
Since we do not focus on the thresholding technique, in practice, the threshold for detecting anomaly from the anomaly score can be selected by the best F1-score. <br>
The detailed testing results can be found in the directory named "log_tester".

## Evaluation results
The detailed resuts of evaluation are in the directory named "log_evaluator". <br>
We obtain all F1-score by enumerating all thresholds and use the best F1-score as the final score.


# Exp_datasets
Three multivariate CDN multivariate KPI datasets，including the dataset of VOD1, VOD2 and LIVE, are used in our work. Each of them spans 78, 64 and 54 days, respectively.  Among them, VOD1 and VOD2 correspond to two video-on-demand websites while the other corresponds to a live streaming website.

## Dataset Information

### Basic Statistics

Statistics | VoD1 | VoD2 | Live
--- | --- | --- | ---
Number of KPIs | 24 | 16 | 48
Durations (day) | 78 | 64 | 54
Granularity (min) | 5 | 1 | 5
Number of points | 22,356 | 91,507 | 15,617
Number of anomaly segments | 7 | 5 | 6
Anomaly ratio (%) | 1.6 | 0.434 | 1.24
Train period | 1 ∼ 10,656 | 1 ~ 51,336 | 1 ~ 7,808
Test period | 10,657 ∼ 22,356 | 51,337 ~ 91,507 | 7,809 ~ 15,617

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

#### Some notes on data
We notice that the KPIs data file and the corresponding ground-truth file cannot be downloaded due to each of *the file is too big to be anonymized (beyond 1MB, Github limit).* <br>
Thus we split the data and put them in the directory name "data_preprocess/data_n_kpi". <br>
Please reconstruct the data in the same format as the described above and put them in "data_preprocess/data" if you want to reproduce some experiments based on these data. <br>
We are sorry for the inconvenience of testing SDFVAE for these reasons.

### Public Dataset 
The public dataset (SMD) used in our evaluation experiments as well as its detailed description can be found in web sites:
https://github.com/NetManAIOps/OmniAnomaly

In order to make it easy for reviewers to run our code on SMD, we select 2 of 28 namely "machine-1-1.txt" and "machine-1-5.txt" as an example. We add the timestamp to the two datasets to create the format required by our data preprocessing code. Please don not forget to add the timestamp if you want to test SDFVAE on others datasets. *It should be noted that SDFVAE never utilise any information of these timestamps to improve its performance.* 



**Note that all KPIs are normalized and we omitted the real name of each KPI for confidentiality, but this does not affect the accuracy of the evaluation experiments.**

