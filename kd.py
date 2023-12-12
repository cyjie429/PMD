import glob
import json
import torch
import shutil
import torch.nn as nn
import torch.utils.data
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from doc import Dataset, collate, load_data
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from loader import TeacherModelLoader
from student_model import StudentModel
import evaluate

from dict_hub import get_entity_dict, EntityDict
entity_dict: EntityDict = get_entity_dict()

class DistilledTrainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node

        # create model
        logger.info("=> creating model")
        self.teacher = TeacherModelLoader()
        self.teacher.load(ckt_path=args.eval_model_path)
        self._freeze_parameters(self.teacher.model)
        self.student = StudentModel(self.args)
        self._setup_training()

        # define loss function (criterion) and optimizer
        self.criterion_hard = nn.CrossEntropyLoss().cuda()
        self.criterion_soft = nn.MSELoss().cuda()
        self.optimizer = AdamW([p for p in self.student.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        
        report_num_trainable_parameters(self.teacher.model)

        # "${DATA_DIR}/train.txt.json"
        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)
        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)
        
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None


    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)


    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        # metric_dict = self.eval_epoch(epoch)
        is_best = True
        # if is_best:
        #     self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.student.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        entity_tensor = evaluate.predict_by_entities(self.student, entity_dict.entity_exs)
        forward_metrics = evaluate.eval_single_direction(self.student,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True)
        backward_metrics = evaluate.eval_single_direction(self.student,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False)
        metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}

        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metrics)))
        return metrics

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            self.student.train()
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])
            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    s_outputs = self.student(**batch_dict)
            else:
                s_outputs = self.student(**batch_dict)
            s_hr_mask_output = s_outputs['hr_mask_output']
            s_tail_mask_output = s_outputs['tail_mask_output']
            s_outputs = get_model_obj(self.student).compute_logits(output_dict=s_outputs, batch_dict=batch_dict)

            s_outputs = ModelOutput(**s_outputs)
            s_logits, s_labels = s_outputs.logits, s_outputs.labels


            self.teacher.model.eval()
            t_outputs = self.teacher.model(**batch_dict)
            t_hr_mask_output = t_outputs['hr_mask_output']
            t_tail_mask_output = t_outputs['tail_mask_output']
            t_outputs = get_model_obj(self.teacher.model).compute_logits(output_dict=t_outputs, batch_dict=batch_dict)
            t_outputs = ModelOutput(**t_outputs)
            t_logits, t_labels = t_outputs.logits, t_outputs.labels

            loss_hard = self.criterion_hard(s_logits, s_labels)
            # tail -> head + relation
            loss_hard += self.criterion_hard(s_logits[:, :batch_size].t(), s_labels)

            # loss_hard += self.criterion_hard(s_logits[:, :batch_size].t(), s_labels)
            loss_global_soft = self.criterion_soft(s_logits, t_logits)

            loss_gen = self.criterion_soft(s_hr_mask_output, t_hr_mask_output)
            loss_gen += self.criterion_soft(s_tail_mask_output, t_tail_mask_output)


            loss = (1-self.args.alpha-self.args.beta) * loss_hard +  self.args.alpha * loss_global_soft + self.args.beta * loss_gen

            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.student = torch.nn.DataParallel(self.student).cuda()
            self.teacher.model = torch.nn.DataParallel(self.teacher.model).cuda()
        elif torch.cuda.is_available():
            self.student.cuda()
            self.teacher.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)

    def _freeze_parameters(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False