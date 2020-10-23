import time, datetime
import os
class IntervalAnomaly:
    def __init__(self):
        self.isLastIntervalAnomaly = False
        self.timeWindow = []
        self.anomalyWindow = []
        self.anomalyIntervalStart = ""
        self.anomalyIntervalEnd = "20180101000001"
        self.windowsize = 3 
        self.IntervalAnomalyThreshold = 1 
        self.IntervalMergeThreshold = 300 

    def IntervalAnomalyDetect(self, realtime, isAnomaly, anomaly_intervals):
        '''
        params:
                realtime: string, as "20180101000001" total 14 bits
                isAnomaly: bool, True (represents Anomaly) or False (represents not Anomaly)
        return:
                None
        '''
        if len(self.timeWindow) < self.windowsize - 1:
            self.timeWindow.append(realtime)
            self.anomalyWindow.append(isAnomaly)
            return
        elif len(self.timeWindow) == self.windowsize - 1:
            self.timeWindow.append(realtime)
            self.anomalyWindow.append(isAnomaly)
        else:
            self.timeWindow.pop(0)
            self.timeWindow.append(realtime)
            self.anomalyWindow.pop(0)
            self.anomalyWindow.append(isAnomaly)

        anomalyCount = 0
        for i in range(self.windowsize):
            if self.anomalyWindow[i]:
                anomalyCount += 1
        if anomalyCount >= self.IntervalAnomalyThreshold:
            if self.isLastIntervalAnomaly == False:
                current_array = time.strptime(self.timeWindow[0], "%Y%m%d%H%M%S")
                last_array = time.strptime(self.anomalyIntervalEnd, "%Y%m%d%H%M%S")
                current_timestamp = int(time.mktime(current_array))
                last_timestamp = int(time.mktime(last_array))
                if (current_timestamp - last_timestamp) <= self.IntervalMergeThreshold:
                    first_anomaly_idx = self.anomalyWindow.index(True)
                    last_anomaly_idx = -self.anomalyWindow[::-1].index(True)-1
                    self.anomalyIntervalEnd = self.timeWindow[self.windowsize + last_anomaly_idx]
                    anomaly_intervals.pop(-1)
                    anomaly_intervals.append([self.anomalyIntervalStart,self.anomalyIntervalEnd])
                else:
                    first_anomaly_idx = self.anomalyWindow.index(True)
                    last_anomaly_idx = -self.anomalyWindow[::-1].index(True)-1
                    self.anomalyIntervalStart = self.timeWindow[first_anomaly_idx]
                    self.anomalyIntervalEnd = self.timeWindow[self.windowsize + last_anomaly_idx]
                    anomaly_intervals.append([self.anomalyIntervalStart, self.anomalyIntervalEnd])
                self.isLastIntervalAnomaly = True
            else:
                first_anomaly_idx = self.anomalyWindow.index(True)
                last_anomaly_idx = -self.anomalyWindow[::-1].index(True)-1
                self.anomalyIntervalEnd = self.timeWindow[self.windowsize + last_anomaly_idx]
                anomaly_intervals.pop(-1)
                anomaly_intervals.append([self.anomalyIntervalStart,self.anomalyIntervalEnd])
        else:
            if self.isLastIntervalAnomaly:
                self.isLastIntervalAnomaly = False
