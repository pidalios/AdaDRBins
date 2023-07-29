import torch
import math
import numpy as np

class Metrics():
    def __init__(self):
        self.loss = 0
        self.time = 0
        self.max_fps = 0
        self.avg_fps = 0
        self.rmse = 0
        self.absrel = 0
        self.abs_diff = 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, loss, time, max_fps, avg_fps, rmse, absrel, delta1, delta2, delta3):
        self.loss = loss
        self.time = time
        self.max_fps = max_fps
        self.avg_fps = avg_fps
        self.rmse = rmse
        self.absrel = absrel
        # self.abs_diff = abs_diff
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3


    def evaluate(self, output, target, loss, time):
        valid_mask = target > 0
        output = output[valid_mask]
        target = target[valid_mask]
        abs_diff = (output - target).abs()

        mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(mse)
        self.absrel = float((abs_diff / target).mean())


        maxRatio = torch.max(output/target, target/output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.loss = loss
        self.time = time



class AverageMetrics():
    def __init__(self):
        self.reset()
        self.count = 0

    def reset(self):
        self.sum_loss = 0
        self.sum_time = 0
        self.sum_fps = 0
        self.max_fps = 0
        # self.sum_time = []
        self.sum_rmse = 0
        self.sum_absrel = 0
        # self.sum_abs_diff = 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, metrics):
        self.count += 1
        self.sum_loss += metrics.loss
        self.sum_time += metrics.time
        fps = 1 / metrics.time
        if fps > self.max_fps:
            self.max_fps = fps
            # print(fps)
        self.sum_fps += fps
        # print(len(self.sum_time)/sum(self.sum_time))
        # if (metrics.time > self.t):
            # print(metrics.time)
            # self.t = metrics.time
        # print(metrics.time)
        # print(self.count)
        # print(self.sum_time)
        
        self.sum_rmse += metrics.rmse
        self.sum_absrel += metrics.absrel
        # self.sum_abs_diff += metrics.abs_diff

        self.sum_delta1 += metrics.delta1
        self.sum_delta2 += metrics.delta2
        self.sum_delta3 += metrics.delta3


    def average(self):
        avg = Metrics()
        avg.update(
                self.sum_loss/self.count, 
                self.sum_time/self.count, 
                self.max_fps, 
                self.sum_fps/self.count, 
                self.sum_rmse/self.count, 
                self.sum_absrel/self.count, 
                # self.sum_abs_diff/self.count, 
                self.sum_delta1/self.count, 
                self.sum_delta2/self.count, 
                self.sum_delta3/self.count
                )
        return avg


