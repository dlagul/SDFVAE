# SDFVAE

# Exp_datasets
Two multivariate time series (structured CDN KPIs) datasets are used in our work (folders named dataset1 and dataset2) which span 78 days and 64 days respectively. <br> 
These two datasets are collected from two different provincial-level CDN edge sites of a top ISP-operated CDN. <br>
There are 7 abnormal sequences in dataset 1 and 5 abnormal sequences in dataset 2. <br>
These labelled abnormal sequences are confirmed by human operators. So they can be considered as the ground truth. <br>
The KPIs in the datasets include the out-bound traffic and in-bound traffic of CDN servers, cache hit ratio, average bitrate, and so on. <br>
For privacy reasons, these KPIs are anonymized and normalized. <br>


## Dataset Information

### Dataset1
#### Basic Statistics

Statistics | dataset1
--- | ---
Number of KPIs | 24
Durations (day) | 78
Granularity (min) | 5
Number of points | 22,356
Number of anomaly sequences | 7
Anomaly ratio (%) | 1.6
Train period | 1 ∼ 10,656
Test period | 10,657 ∼ 22,356

#### Data format
There are 24 CSV files and each file corresponds to a KPI. <br>
The CSV has the following format: <br>
* First column is the timestamp <br>
* Second column is the value <br>
* Third column is the label. 0 for normal and 1 for abnormal <br>

Timestamp | Value | Label
--- | --- | ---
20181001000500 | 0.46444977152338346 | 0
20181001001000 | 0.4423121844530866 | 0
20181001001500 | 0.4186436700242946 | 0
20181001002000 | 0.39892597116922146 | 1
20181001002500 | 0.37977501905111494 | 1
20181001003000 | 0.36615750635812155 | 1

### Dataset2

#### Basic Statistics

Statistics | dataset2
--- | ---
Number of KPIs | 16
Durations (day) | 64
Granularity (min) | 1
Number of points | 91,507
Number of anomaly sequences | 5
Anomaly ratio (%) | 0.32
Train period | 1 ∼ 51,336
Test period | 51,337 ∼ 91,507

#### Data format
There are 2 CSV files in the folder named "dataset2". <br>
The file named "dataset2.csv" is the full data, and "dataset2_sample.csv" is the sample which contains 1000 records. <br>
The CSV has the following format: <br>
* First column is the timestamp <br>
* 2nd~17th columns are the KPI values which correspond to 16 KPIs <br>
* The last column is the label. 0 for normal and 1 for abnormal <br>

Timestamp | Kpi1 | ... | Kpi16 | Label
--- | --- | --- | --- | ---
20190903000200 | 0.46444977152338346 | ... | -0.588230235 | 0 |
20190903000300 | 0.4423121844530866 | ... | -0.595955299 | 0 |
20190903000400 | 0.4186436700242946 | ... | -0.600299795 | 0 |
20190903000500 | 0.39892597116922146 | ... | -0.604815951 | 1 |
20190903000600 | 0.37977501905111494 | ... | -0.610974025 | 1 |
20190903000700 | 0.36615750635812155 | ... | -0.616264816 | 1 |

### Public Dataset 
The public dataset (SMD) used in our evaluation experiments as well as its detailed description can be found in web sites:
https://github.com/NetManAIOps/OmniAnomaly

For simplicity, we select 2 of 28 machine data namely "machine-1-2.txt" and "machine-1-3.txt" to conduct evaluation experiments.

**Note that all KPIs are normalized and we omitted the real name of each KPI for confidentiality, but this does not affect the accuracy of the evaluation experiments.**
