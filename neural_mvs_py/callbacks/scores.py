
import torchnet
import numpy as np
import torch
import csv

            
class Scores():
    def __init__(self):

        self.psnr_acum=0
        self.nr_times_accumulated=0
        self.cur_epoch_nr=0
        #things fro keeping track of the best starts and the best epoch
        self.best_avg_psnr=0
        self.best_epoch_nr=0 #epoch idx where the best psnr was found

        #attempt 2
        self.clear()


    #adapted from https://github.com/NVlabs/splatnet/blob/f7e8ca1eb16f6e1d528934c3df660bfaaf2d7f3b/splatnet/semseg3d/eval_seg.py
    def accumulate_scores(self, psnr):
        self.psnr_acum+=psnr
        self.nr_times_accumulated+=1
        
    #compute all the starts that you may need, psnr, ssim, etc . TODO implement ssim and other metrics
    def compute_stats(self, print_stats=False):
        
        if(self.nr_times_accumulated==0):
            avg_psnr=0
        else:
            avg_psnr=self.psnr_acum/self.nr_times_accumulated
        
        if print_stats:
            print("average psnr is", avg_psnr  )
        return avg_psnr


    def avg_psnr(self, print_stats=False):
        avg_psnr=self.compute_stats(print_stats=print_stats)
        return avg_psnr

    def update_best(self):
        avg_psnr=self.compute_stats(print_stats=False)
        if avg_psnr>self.best_avg_psnr:
            self.best_avg_psnr=avg_psnr
            self.best_epoch_nr=self.cur_epoch_nr




    def show(self, epoch_nr):

        self.compute_stats(print_stats=True)
        
    #clear once, when you want to start completely fresh, with no knowledge of the best stats
    def clear(self):

        self.psnr_acum=0
        self.nr_times_accumulated=0
        self.best_avg_psnr=0
        self.best_epoch_nr=0
       
    #to be called every epoch, to clear the statistics for that epoch but keep in mind the knowledge of the best stats
    def start_fresh_eval(self, cur_epoch_nr):
        self.psnr_acum=0
        self.nr_times_accumulated=0
        self.cur_epoch_nr=cur_epoch_nr
       

    def write_stats_to_csv(self,filename):
        avg_psnr=self.avg_psnr(print_stats=False)
        w = csv.writer(open(filename, "w"))
        w.writerow(["avg_psnr", avg_psnr])
        w.writerow(["best_avg_psnr", self.best_avg_psnr])
        w.writerow(["best_epoch_nr", self.best_epoch_nr])

  
