import os
import sys
import abc
import math
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from src.option import default_hparas
from src.util import human_format, Timer

class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
    Arguments
        config - yaml-styled config
        paras  - argparse outcome
    '''
    def __init__(self, config, paras, mode):
        # General Settings
        self.config = config
        self.paras = paras
        self.mode = mode
        for k,v in default_hparas.items():
            setattr(self,k,v)
        self.device = torch.device('cuda') if self.paras.gpu and torch.cuda.is_available() else torch.device('cpu')
        self.amp = paras.amp

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = paras.config.split('/')[-1].replace('.yaml','') # By default, exp is named after config file
            if mode == 'train':
                self.exp_name += '_sd{}'.format(paras.seed)

        # Plugin list
        self.emb_decoder = None

        if mode == 'train':
            # Filepath setup
            os.makedirs(paras.ckpdir, exist_ok=True)
            self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
            os.makedirs(self.ckpdir, exist_ok=True)

            # Logger settings
            self.logdir = os.path.join(paras.logdir,self.exp_name)
            self.log = SummaryWriter(self.logdir, flush_secs = self.TB_FLUSH_FREQ)
            self.timer = Timer()

            # Hyperparameters
            self.step = 0
            self.valid_step = config['hparas']['valid_step']
            self.max_step = config['hparas']['max_step']
            if config['hparas'].get('grad_clip'):
                self.GRAD_CLIP = float(config['hparas']['grad_clip'])
            
            self.verbose('Exp. name : {}'.format(self.exp_name))
            self.verbose('Loading data... large corpus may took a while.')
            
        elif mode == 'test':
            # Output path
            os.makedirs(paras.outdir, exist_ok=True)
            self.ckpdir = os.path.join(paras.outdir,self.exp_name)

            # Load training config to get acoustic feat, text encoder and build model
            self.src_config = yaml.load(open(config['src']['config'],'r'), Loader=yaml.FullLoader)
            self.paras.load = config['src']['ckpt']

            self.verbose('Evaluating result of tr. config @ {}'.format(config['src']['config'])) 

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load option is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load(self.paras.load, map_location=self.device if self.mode=='train' else 'cpu')
            self.model.load_state_dict(ckpt['model'])
            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(ckpt['emb_decoder'])
            #if self.amp:
            #    amp.load_state_dict(ckpt['amp'])
            # Load task-dependent items
            if self.mode == 'train':
                if cont:
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(self.paras.load,self.step))
            else:
                for k,v in ckpt.items():
                    if type(v) is float:
                        metric, score = k,v
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
                self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(self.paras.load,metric,score))

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            if type(msg)==list:
                for m in msg:
                    print('[INFO]',m.ljust(100))
            else:
                print('[INFO]',msg.ljust(100))

    def progress(self,msg):
        ''' Verbose function for updating progress on stdout (do not include newline) '''
        if self.paras.verbose:
            sys.stdout.write("\033[K") # Clear line
            print('[{}] {}'.format(human_format(self.step),msg),end='\r')
    
    def write_log(self,log_name,log_value):
        '''
        Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable 
            log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
        '''
        if type(log_value) is dict:
            log_value = {key:val for key, val in log_value.items() if (val is not None and not math.isnan(val))}
        if log_value is None:
            pass
        elif len(log_value)>0:
            # ToDo : support all types of input
            if 'align' in log_name or 'spec' in log_name or 'hist' in log_name:
                img, form = log_value
                self.log.add_image(log_name,img, global_step=self.step, dataformats=form)
            elif 'code' in log_name:
                self.log.add_embedding(log_value[0], metadata=log_value[1], tag=log_name, global_step=self.step)
            elif 'wave' in log_name:
                signal, sr = log_value
                self.log.add_audio(log_name, torch.FloatTensor(signal).unsqueeze(0), self.step, sr)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_value, self.step)
            else:
                self.log.add_scalars(log_name,log_value,self.step)

    def save_checkpoint(self, f_name, metric, score):
        '''' 
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            metric: score
        }
        # Additional modules to save
        #if self.amp:
        #    full_dict['amp'] = self.amp_lib.state_dict()
        if self.emb_decoder is not None:
            full_dict['emb_decoder'] = self.emb_decoder.state_dict()

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".\
                                       format(human_format(self.step),metric,score,ckpt_path))

    def enable_apex(self):
        if self.amp:
            # Enable mixed precision computation (ToDo: Save/Load amp)
            from apex import amp
            self.amp_lib = amp
            self.verbose("AMP enabled (check https://github.com/NVIDIA/apex for more details).")
            self.model, self.optimizer.opt = self.amp_lib.initialize(self.model, self.optimizer.opt, opt_level='O1')


    # ----------------------------------- Abtract Methods ------------------------------------------ #
    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup (e.g. self.tr_set, self.dev_set)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup (e.g. self.l2_loss)
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
                init. w/ self.optimizer = src.Optimizer(self.model.parameters(),**self.config['hparas'])
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        '''
        raise NotImplementedError


