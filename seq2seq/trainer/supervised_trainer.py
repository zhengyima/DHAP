from __future__ import division
import os
import logging
import random
import time
# import horovod.torch as hvd
import torch
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
import json
class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        model_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, 
                 model_dir='experiment',
                 best_model_dir='experiment/best',
                 loss=NLLLoss(), 
                 batch_size=64, 
                 checkpoint_every=100, 
                 print_every=100, 
                 max_epochs=5,
                 max_steps=10000, 
                 max_checkpoints_num=5, 
                 best_ppl=100000.0,
                 device=None,
                 multi_gpu=False,
                 logger=None,src_idx2word=None,tgt_idx2word=None,tgt_vocab=None):
        self._trainer = "Simple Trainer"
        self.loss = loss
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.best_ppl = best_ppl
        self.max_checkpoints_num = max_checkpoints_num
        self.device = device
        self.multi_gpu = multi_gpu
        
        self.src_idx2word = src_idx2word
        self.tgt_idx2word = tgt_idx2word
        self.tgt_vocab = tgt_vocab
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, device=device, tgt_idx2word=tgt_idx2word, tgt_vocab=tgt_vocab)

        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.getcwd(), model_dir)
        self.model_dir = model_dir
        
        if not os.path.isabs(best_model_dir):
            best_model_dir = os.path.join(os.getcwd(), best_model_dir)
        self.best_model_dir = best_model_dir

        if not multi_gpu or hvd.rank() == 0:
            if not os.path.exists(self.best_model_dir): os.makedirs(self.best_model_dir)
            if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        self.model_checkpoints = []
        self.best_model_checkpoints = []

        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def save_model(self, model, steps, dev_ppl=None, allresult=None):
        model_fn = f"{steps}.pt"
        model_fp = os.path.join(self.model_dir, model_fn)

        # save model checkpoints
        while len(self.model_checkpoints) >= self.max_checkpoints_num:
            os.system(f"rm {self.model_checkpoints[0]}")
            self.model_checkpoints = self.model_checkpoints[1:]
        torch.save(model.state_dict(), model_fp)
        self.model_checkpoints.append(model_fp)

        # update checkpoints file
        with open(os.path.join(self.model_dir, "checkpoints"), 'w') as f:
            f.write('\n'.join(self.model_checkpoints[::-1]))

        if not dev_ppl: return None
        # save best model checkpoints
        if dev_ppl < self.best_ppl:


            self.logger.info(f"Best model dev ppl {dev_ppl}.")
            self.best_ppl = dev_ppl
            while len(self.best_model_checkpoints) >= self.max_checkpoints_num:
                os.system(f"rm {self.best_model_checkpoints[0]}")
                self.best_model_checkpoints = self.best_model_checkpoints[1:]
            
            best_model_fp = os.path.join(self.best_model_dir, model_fn)
            os.system(f"cp {model_fp} {best_model_fp}")
            self.best_model_checkpoints.append(best_model_fp)

            if allresult != None:

                f = open(best_model_fp+".txt","w")
                for bd in allresult:
                    post = bd['post']
                    answer = bd['answer']
                    result = bd['result']
                    # for i in range(len(post)):
                        # print(post[i])
                    # for w in post:
                    #     print(w.item(),self.src_idx2word[w.item()])
                    post_words = [self.src_idx2word[w.item()] for w in post]
                    answer_words = [self.tgt_idx2word[w.item()] for w in answer]
                    result_words = [self.tgt_idx2word[w.item()] for w in result]
                    f.write(json.dumps({"post":post_words,"answer":answer_words,"result":result_words},ensure_ascii=False)+"\n")
                f.close()



        else:
            self.logger.info(f"Current learning rate: {self.optimizer.optimizer.param_groups[0]['lr']}")


    def _train_batch(self, input_variable, input_lengths, target_variable, r_history_idx, r_history_idx_pos,
                r_mask,  r_mask_pos,  p_pos, p_history_idx, p_history_idx_pos, model, teacher_forcing_ratio):
        loss = self.loss
        device = self.device
        # Forward propagation
        # print("i_v",input_variable)
        # print("i_l",input_lengths)
        # print("t_v",target_variable)
        # print(torch.max(r_history_extend_vocab).item(), len(self.tgt_idx2word) + r_max_oov.item())
        # assert torch.max(r_history_extend_vocab).item() < len(self.tgt_idx2word) + r_max_oov.item()
        # assert torch.max(r_extend_vocab).item() < len(self.tgt_idx2word) + r_max_oov.item()
        # print(torch.max(r_extend_vocab).item(), len(self.tgt_idx2word) + r_max_oov.item())
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
        r_history_idx, r_history_idx_pos,r_mask,  r_mask_pos, p_pos, p_history_idx, p_history_idx_pos, teacher_forcing_ratio=teacher_forcing_ratio, istest=False)
        # print("decoder_outputs", decoder_outputs)
        # Get loss
        loss.reset()
        # weights = torch.ones(len(self.tgt_idx2word)+r_max_oov[0])
        # weights[self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] = 0
        # weights = weights.to(device)
        # loss.criterion = torch.nn.NLLLoss(weight=weights, reduction=loss.reduction).to(device)
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            # print("step_output",step_output.contiguous().view(batch_size, -1).size())
            # print("target_variable",target_variable[:, step + 1])
            # loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
            
            
            # print(step_output.size())
            # print(step_output)
            # print("before loss eval_batch")
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
            # print("after loss eval batch")
        # Backward propagation
        model.zero_grad()
        
        # get_loss_ = loss.get_loss()
        # print("loss",get_loss_)
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        device = self.device
        log = self.logger
        max_epochs = self.max_epochs
        max_steps = self.max_steps
        multi_gpu = self.multi_gpu

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        step = 0
        steps_per_epoch = len(data)
        start_epoch = (start_step - step) // steps_per_epoch
        step = start_epoch * steps_per_epoch
        for batch in data:
            if step >= start_step: break
            step += 1
        if start_epoch or start_step:
            if not multi_gpu or hvd.rank() == 0: log.info(f"Resume from Epoch {start_epoch}, Step {start_step}")

        for epoch in range(start_epoch, max_epochs):
            model.train(True)
            for p_idx, p_idx_len, r_idx, r_idx_len, r_history_idx, r_history_idx_pos, r_mask, r_mask_pos, p_pos, p_history_idx, p_history_idx_pos in data:
                step += 1
                src_variables = p_idx.to(device)
                tgt_variables = r_idx.to(device)
                src_lens = p_idx_len.view(-1).to(device)
                tgt_lens = r_idx_len.view(-1).to(device)
                r_history_idx = r_history_idx.to(device)
                r_history_idx_pos = r_history_idx_pos.to(device)
                r_mask = r_mask.to(device)
                r_mask_pos = r_mask_pos.to(device)
                p_pos = p_pos.to(device)
                p_history_idx = p_history_idx.to(device)
                p_history_idx_pos = p_history_idx_pos.to(device)
                # r_history_extend_vocab = r_history_extend_vocab.to(device)
                # oovs = oovs.to(device)
                # r_history_mask = r_history_mask.to(device)
                # r_max_oov = max()
                
                # r_extend_vocab = r_extend_vocab.to(device)
                # print(r_max_oov)
                
                # r_max_oov = max([len(oov) for oov in oovs])
                # print([len(oov) for oov in oovs])
                # print(r_max_oov)
                # r_max_oov = torch.LongTensor([r_max_oov]).to(device)
                # print(r_max_oov)
                # extra_zeros = torch.zeros([output_flatten.size(0),r_max_oov[0]]).to(1)+1e-10



                # print(src_variables, src_lens, tgt_variables)
                # exit(0)

                loss = self._train_batch(src_variables, src_lens.tolist(), tgt_variables, r_history_idx, r_history_idx_pos,
                r_mask,  r_mask_pos,  p_pos, p_history_idx, p_history_idx_pos, model, teacher_forcing_ratio)

                # print("loss:",loss)
                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = f"Process {100.0*(step%steps_per_epoch)/steps_per_epoch:.2f}% of Epoch {epoch}, Total step {step}, Train {self.loss.name} {print_loss_avg:.4f}" 
                    print(log_msg)
                    if not multi_gpu or hvd.rank() == 0:
                        log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0:
                    dev_loss = None
                    if dev_data is not None:
                        dev_loss, accuracy, allresult = self.evaluator.evaluate(model, dev_data, self.src_idx2word, self.tgt_idx2word)
                        self.optimizer.update(dev_loss, epoch)
                        log_msg = f"Dev {self.loss.name}: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"

                        print(log_msg)
                        if not multi_gpu or hvd.rank() == 0:
                            log.info(log_msg)
                        model.train(mode=True)
                    if not multi_gpu or hvd.rank() == 0:
                        self.save_model(model, step, dev_ppl=dev_loss, allresult=allresult)

                if step >= max_steps:
                    break

            if step >= max_steps:
                if not multi_gpu or hvd.rank() == 0:
                    log.info(f"Finish max steps {max_steps} at Epoch {epoch}.")
                break

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = f"Finished Epoch {epoch}, Train {self.loss.name} {epoch_loss_avg:.4f}"
            allresult = None
            if dev_data is not None:
                dev_loss, accuracy, allresult  = self.evaluator.evaluate(model, dev_data, self.src_idx2word, self.tgt_idx2word)
                self.optimizer.update(dev_loss, epoch)
                log_msg += f", Dev {self.loss.name}: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)
            if not multi_gpu or hvd.rank() == 0:
                self.save_model(model, step, dev_ppl=dev_loss, allresult=allresult)
                log.info(log_msg)
                log.info(f"Finish Epoch {epoch}, Total steps {step}.")

    def train(self, model, data, start_step=0, dev_data=None, optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        
        if optimizer is None:
            optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
        self.optimizer = optimizer

        if not self.multi_gpu or hvd.rank() == 0:
            self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, start_step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
