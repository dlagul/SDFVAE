import argparse
import os
import numpy as np
from get_interval_anomaly import IntervalAnomaly
import time
from logger import Logger

class Evaluator():
    def __init__(self, anomaly_score_label_file, th_range = [-10,10], th_step = 1, log_path='', log_file=''):
        self.anomaly_score_label_file = anomaly_score_label_file
        self.th_range = th_range
        self.th_step = th_step
        self.log_path = log_path
        self.log_file = log_file
        self.label = []
        self.eval_metrics = {}
        self.best_eval_metrics = ''
        self.f1_best = 0
        self.pr_auc = 0
        self.ground_truth_anomaly_intervals = []
        self.detected_anomaly_intervals = []
        self.reconstructed_anomaly_intervals = []
        self.detected_result = []
        self.timestamp_detected_result = []
        self.logger = Logger(self.log_path, self.log_file)

    def get_ground_truth_anomaly_intervals(self, timestamp_anomalyscore_label):
        ground_truth_anomaly_intervals = []
        IA_ground_truth = IntervalAnomaly()
        for i in range(len(timestamp_anomalyscore_label[0])):
            if timestamp_anomalyscore_label[2][i] == "Anomaly":
                isAnomaly = True
            else:
                isAnomaly = False
            IA_ground_truth.IntervalAnomalyDetect(timestamp_anomalyscore_label[0][i], isAnomaly, ground_truth_anomaly_intervals)
        del IA_ground_truth
        return ground_truth_anomaly_intervals

    def get_detected_anomaly_intervals(self, th, timestamp_anomalyscore_label):
        anomaly_detected = []
        detected_anomaly_intervals = []
        for idx in range(len(timestamp_anomalyscore_label[0])):
            if float(timestamp_anomalyscore_label[1][idx]) <= th:
                anomaly_detected.append(True)
            else:
                anomaly_detected.append(False)
        IA_detected = IntervalAnomaly()
        for k in range(len(anomaly_detected)):
            IA_detected.IntervalAnomalyDetect(timestamp_anomalyscore_label[0][k],anomaly_detected[k],detected_anomaly_intervals)
        del IA_detected
        return detected_anomaly_intervals

    def get_reconstruct_detected_anomaly_intervals(self, timestamp_anomalyscore_label, 
                                                   ground_truth_anomaly_intervals, detected_anomaly_intervals):
        detected_anomaly_intervals_tmp = detected_anomaly_intervals
        for i in range(len(timestamp_anomalyscore_label[0])):
            current_timestamp = int(time.mktime(time.strptime(timestamp_anomalyscore_label[0][i], "%Y%m%d%H%M%S")))
            for j in range(len(ground_truth_anomaly_intervals)):
                start_gt = int(time.mktime(time.strptime(ground_truth_anomaly_intervals[j][0], "%Y%m%d%H%M%S")))
                end_gt = int(time.mktime(time.strptime(ground_truth_anomaly_intervals[j][1], "%Y%m%d%H%M%S")))
                if current_timestamp >= start_gt and current_timestamp <= end_gt:
                    for k in range(len(detected_anomaly_intervals)):
                        start_dt = int(time.mktime(time.strptime(detected_anomaly_intervals[k][0], "%Y%m%d%H%M%S")))
                        end_dt = int(time.mktime(time.strptime(detected_anomaly_intervals[k][1], "%Y%m%d%H%M%S")))
                        if current_timestamp >= start_dt and current_timestamp <= end_dt:
                            detected_anomaly_intervals_tmp[k] = ground_truth_anomaly_intervals[j]
        return detected_anomaly_intervals_tmp

    def get_detected_result(self, reconstruct_detected_anomaly_intervals,timestamp_anomalyscore_label):
        timestamp_detected_result = []
        detected_result = []
        for i in range(len(timestamp_anomalyscore_label[0])):
            current_timestamp = int(time.mktime(time.strptime(timestamp_anomalyscore_label[0][i], "%Y%m%d%H%M%S")))
            flag = False
            for j in range(len(reconstruct_detected_anomaly_intervals)):
                start_gt = int(time.mktime(time.strptime(reconstruct_detected_anomaly_intervals[j][0], "%Y%m%d%H%M%S")))
                end_gt = int(time.mktime(time.strptime(reconstruct_detected_anomaly_intervals[j][1], "%Y%m%d%H%M%S")))
                if current_timestamp >= start_gt and current_timestamp <= end_gt:
                    flag = True
                    break
            timestamp_detected_result.append(str(timestamp_anomalyscore_label[0][i])+","+str(flag))
            detected_result.append(flag)
        return timestamp_detected_result,detected_result

    def get_metrics(self, label, detected_result):
        assert len(label) == len(detected_result)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(label)):
            if label[i]==True and detected_result[i] == True:
                TP = TP+1
            elif label[i]==True and detected_result[i] == False:
                FN = FN+1
            elif label[i]==False and detected_result[i] == False:
                TN = TN+1
            elif label[i]==False and detected_result[i] == True:
                FP = FP+1
        if TP+FP-0 != 0:
            precision = TP/(TP+FP)
        else:
            precision = 0
        if TP+FN-0 != 0:
            recall = TP/(TP+FN)
            tpr = TP/(TP+FN)
            fnr = FN/(TP+FN)
        else:
            recall = 0
            fnr = 0
            tpr = 0
        if FP+TN-0 != 0:
            tnr = TN/(FP+TN)
            fpr = FP/(FP+TN)
        else:
            tnr = 0
            fpr = 0
        if precision+recall-0 != 0:
            f1 = 2*precision*recall/(precision+recall)
        else:
            f1 = 0
        return TP, FN, TN, FP, precision, recall, f1, fpr, tpr

    def get_label(self,timestamp_anomalyscore_label):
        label = []
        for idx in range(len(timestamp_anomalyscore_label[2])):
            if timestamp_anomalyscore_label[2][idx] == "Anomaly":
                label.append(True)
            else:
                label.append(False)
        return label


    def perform_evaluating(self):
        timestamp_anomalyscore_label1 = np.loadtxt(self.anomaly_score_label_file, delimiter=',', dtype=bytes, unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()  
        timestamp_anomalyscore_label3 = [[],[],[]]
        for i in range(len(timestamp_anomalyscore_label2)):
            timestamp_anomalyscore_label3[0].append(timestamp_anomalyscore_label2[i][0])
            timestamp_anomalyscore_label3[1].append(timestamp_anomalyscore_label2[i][1])
            timestamp_anomalyscore_label3[2].append(timestamp_anomalyscore_label2[i][2])
        timestamp_anomalyscore_label = np.array(timestamp_anomalyscore_label3)
        anomaly_score_min = np.min(timestamp_anomalyscore_label[1].astype(float),axis=0)
        anomaly_score_max = np.max(timestamp_anomalyscore_label[1].astype(float),axis=0)
        if self.th_range[1] >= anomaly_score_max:
            if self.th_range[0] >= anomaly_score_min: 
                threshold_candidates = [t for t in np.arange(self.th_range[0],anomaly_score_max,self.th_step)]
            else:
                threshold_candidates = [t for t in np.arange(anomaly_score_min,anomaly_score_max,self.th_step)]
        else:
            if self.th_range[0] >= anomaly_score_min:
                threshold_candidates = [t for t in np.arange(self.th_range[0],self.th_range[1],self.th_step)]
            else:
                threshold_candidates = [t for t in np.arange(anomaly_score_min,self.th_range[1],self.th_step)]
        self.ground_truth_anomaly_intervals = self.get_ground_truth_anomaly_intervals(timestamp_anomalyscore_label)
        fscore = {}
        for th in threshold_candidates:
            self.detected_anomaly_intervals = self.get_detected_anomaly_intervals(th, timestamp_anomalyscore_label)
            self.reconstruct_detected_anomaly_intervals = self.get_reconstruct_detected_anomaly_intervals(timestamp_anomalyscore_label,
                                     self.ground_truth_anomaly_intervals, self.detected_anomaly_intervals)
            self.timestamp_detected_result,self.detected_result = self.get_detected_result(self.reconstruct_detected_anomaly_intervals,
                                                                                      timestamp_anomalyscore_label)
            self.label = self.get_label(timestamp_anomalyscore_label)
            TP, FN, TN, FP, precision, recall, f1, fpr, tpr = self.get_metrics(self.label, self.detected_result)
            fscore[f1] = "th:{}, p:{}, r:{}, f1score:{}, TP:{}, FN:{}, TN:{}, FP:{}, FPR:{}, TPR:{}".format(
                          th, precision, recall, f1,TP, FN, TN, FP,fpr, tpr)
            self.eval_metrics['Th'] = th
            self.eval_metrics['P'] = precision
            self.eval_metrics['R'] = recall
            self.eval_metrics['F1score'] = f1
            self.eval_metrics['TP'] = TP
            self.eval_metrics['FN']= FN
            self.eval_metrics['TN'] = TN
            self.eval_metrics['FP'] = FP
            self.eval_metrics['Fpr'] = fpr
            self.eval_metrics['Tpr'] = tpr   
            self.logger.log_evaluator(self.eval_metrics)    
            # If the recall has been reached to 1.0, we break the loop, due to the best f1-score has been achieved 
            # Since as the threshold increases, recall remains unchanged (1.0), while precision decreases and thus f1-score decreases
            if float(recall) < 1.0:
                continue
            elif float(recall) == 1.0:
                break 

        fscore_sorted_by_key = sorted(fscore.items(), key=lambda d:d[0], reverse = True)
        self.f1_best = fscore_sorted_by_key[0][0] 
        self.best_eval_metrics = fscore_sorted_by_key[0][1]
        self.logger.log_evaluator_re("f1-best is: {}".format(fscore_sorted_by_key[0][0]))
        self.logger.log_evaluator_re("details: {}".format(fscore_sorted_by_key[0][1]))


def main():
 
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--data_nums', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=36)
    parser.add_argument('--l', type=int, default=10)
    parser.add_argument('--n', type=int, default=24)

    # Model options
    parser.add_argument('--s_dims', type=int, default=8)
    parser.add_argument('--d_dims', type=int, default=10)
    parser.add_argument('--conv_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=40)
    parser.add_argument('--enc_dec', type=str, default='CNN')

    # Training options
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--checkpoints_path', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=10)
    parser.add_argument('--log_path', type=str, default='log_evaluator')
    parser.add_argument('--log_file', type=str, default='')

    parser.add_argument('--llh_path', type=str, default='log_tester')
    parser.add_argument('--llh_file', type=str, default='')
   
    parser.add_argument('--th_min', type=float, default=-50)
    parser.add_argument('--th_max', type=float, default=10)
    parser.add_argument('--th_step', type=float, default=0.2)

    args = parser.parse_args()

    if args.llh_file == '':
        args.ll_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_loss.txt'.format(
                        args.s_dims,
                        args.d_dims,
                        args.conv_dims,
                        args.hidden_dims,
                        args.win_size,
                        args.T, 
                        args.l,
                        args.start_epoch)

    if args.log_file == '':
        args.log_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_eval_records'.format(
                         args.s_dims,
                         args.d_dims,
                         args.conv_dims,
                         args.hidden_dims,
                         args.win_size,
                         args.T, 
                         args.l,
                         args.start_epoch)
    
    if not os.path.exists(os.path.join(args.llh_path,args.llh_file)):
        raise ValueError('Unknown anomaly score label file: {}'.format(args.llh_path))
   
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    anomaly_score_label_file = os.path.join(args.llh_path,args.ll_file)  
    evaluator = Evaluator(anomaly_score_label_file, 
                          th_range = [args.th_min,args.th_max], 
                          th_step  = args.th_step, 
                          log_path = args.log_path, 
                          log_file = args.log_file)

    evaluator.perform_evaluating()

if __name__ == '__main__':
    main()

