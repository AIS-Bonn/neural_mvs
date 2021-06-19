from callbacks.callback import *
import os
import torch


class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, loss, psnr, **kwargs):
        phase.iter_nr+=1
        phase.samples_processed_this_epoch+=1
        phase.loss_acum_per_epoch+=loss

        phase.scores.accumulate_scores(psnr)

    def epoch_started(self, phase, **kwargs):
        phase.loss_acum_per_epoch=0.0
        phase.scores.start_fresh_eval(phase.epoch_nr)

    def epoch_ended(self, phase, model, save_checkpoint, checkpoint_path, save_every_x_epoch, **kwargs):
        phase.scores.update_best()

        message_string=""
        message_string+="epoch "+str(phase.epoch_nr)+" phase.grad is " + str(phase.grad)

        #for evaluation phase print the iou
        if not phase.grad:
            avg_psnr=phase.scores.avg_psnr(print_stats=False)
            best_avg_psnr=phase.scores.best_avg_psnr
            message_string+=" best_avg_psnr " + str(best_avg_psnr) + " at epoch " + str(phase.scores.best_epoch_nr) + " avg psnr this epoch " + str(avg_psnr)

        #save the checkpoint of the model if we are in testing mode
        if not phase.grad:
            if save_checkpoint and model is not None and phase.epoch_nr%save_every_x_epoch==0:
                model_name="model_e_"+str(phase.epoch_nr)+"_score_"+str(avg_psnr)+".pt"
                info_txt_name="model_e_"+str(phase.epoch_nr)+"_info"+".csv"
                out_model_path=os.path.join(checkpoint_path, model_name)
                out_info_path=os.path.join(checkpoint_path, info_txt_name)
                torch.save(model.state_dict(), out_model_path)
                phase.scores.write_stats_to_csv(out_info_path)

        print(message_string)

        

        phase.epoch_nr+=1

    def phase_started(self, phase, **kwargs):
        phase.samples_processed_this_epoch=0

    def phase_ended(self, phase, **kwargs):
        pass
        if phase.loader.is_finished():
            print("resetting loader for phase ", phase.name)
            phase.loader.reset()
