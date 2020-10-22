import torchnet
import numpy as np
import torch

node_name="lnn"
port=8097
# logger_iou = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_iou'}, port=port, env='train_'+node_name)

#map a range of values to another range of values for pytorch tensors
def map_range(input, input_start,input_end, output_start, output_end):
    input_clamped=torch.clamp(input, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)


class Vis():
    def __init__(self, env, port):
        self.port=port
        self.env=env
        self.win_id="0"
        # self.win_id=None

        self.name_dict=dict()
        self.logger_dict=dict()
        self.exp_alpha=0.03 #the lower the value the smoother the plot is

    def update_val(self, val, name, smooth):
        if name not in self.name_dict:
            self.name_dict[name]=val
        else:
            if smooth:
                self.name_dict[name]= self.name_dict[name] + self.exp_alpha*(val-self.name_dict[name])
            else: 
                self.name_dict[name]=val
        
        return self.name_dict[name]

    def update_logger(self, x_axis, val, name_window, name_plot):
        if name_window not in self.logger_dict:
            self.logger_dict[name_window]=torchnet.logger.VisdomPlotLogger('line', opts={'title': name_window}, port=self.port, env=self.env, win=self.win_id)
            print("started new line plot on win ", self.logger_dict[name_window].win)

        # print("update_logger val is ", val, "name plot is ", name_plot)
        self.logger_dict[name_window].log(x_axis, val, name=name_plot)

    def log(self, x_axis, val, name_window, name_plot, smooth, show_every=1):
        new_val=self.update_val(val,name_plot, smooth)
        if(x_axis%show_every==0):
            self.update_logger(x_axis, new_val, name_window, name_plot)

    def show_img_from_tensor(self, tensor, name_window, min_val=0.0, max_val=1.0):
        if name_window not in self.logger_dict:
            self.logger_dict[name_window]=torchnet.logger.VisdomLogger('image', opts={'title': name_window}, port=self.port, env=self.env, win=self.win_id)

        img=tensor.cpu().squeeze(0).detach()
        # print("min max", img.min(), img.max())
        img=map_range(img, min_val, max_val, 0.0, 1.0)
        # print("after mapping range min max", img.min(), img.max())

        # print("img has shape ", img.shape)


        self.logger_dict[name_window].log(img)



