from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import json
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process
from muffin.eval.muffin_inference_logp import get_batch_logps, concate_pad, tdpo_get_batch_logps #, add_diffusion_noise


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class MuffinTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        def should_zero_lr(param_name: str):
            if 'beit3' in param_name:
                if '.A' in param_name:
                    return True
                if 'beit3.vision_embed' in param_name:
                    return True
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (p.requires_grad and should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": 0.0,
                "initial_lr": 0.0
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": 0.0,
            },
        ]
        for n, p in model.named_parameters():
            if should_zero_lr(n) and is_main_process():
                pass
                # print(f'Zero LR params: {n}', flush=True)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        self.scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer)
        print(f'LR schduler is ', str(self.scheduler))



class MuffinDPOTrainer(MuffinTrainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):

        data_dict = inputs
        win_reference_vocab_ps = data_dict.pop('win_reference_vocab_ps')
        rej_reference_vocab_ps = data_dict.pop('rej_reference_vocab_ps')
        
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')

        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')

        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')

        ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
        ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
        uncond_ref_win_logp = data_dict.pop('uncond_ref_win_logp')
        uncond_ref_rej_logp = data_dict.pop('uncond_ref_rej_logp')
        ref_win_logp = data_dict.pop('ref_win_logp')
        ref_rej_logp = data_dict.pop('ref_rej_logp')
        ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
        ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
        if self.args.dpo_use_average:
            ref_win_logp = ref_win_avg_logp
            ref_rej_logp = ref_rej_avg_logp

        idx = data_dict.pop('idx', '')
        images = data_dict.pop('images')
        diffusion_image = data_dict.pop('diffusion_image', '')
        random_image = data_dict.pop('random_image', '')
        crop_image = data_dict.pop('crop_image', '')
        rotate_image = data_dict.pop('rotate_image', '')
        if self.args.use_image_type == 'diffusion':
            tmp_image = diffusion_image
        elif self.args.use_image_type == 'black':
            tmp_image = torch.zeros_like(images)
        elif self.args.use_image_type == 'crop':
            tmp_image = crop_image
        elif self.args.use_image_type == 'rotate':
            tmp_image = rotate_image
        elif self.args.use_image_type == 'random':
            tmp_image = random_image

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        concatenated_images = torch.cat([images, images, tmp_image], dim=0)

        win_token_weight = data_dict.pop('win_token_weight')
        rej_token_weight = data_dict.pop('rej_token_weight')
        concatenated_token_weight = data_dict.pop('concatenated_token_weight')


        concatenated_logp, all_position_kl = self.forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        win_reference_vocab_ps,
                                        rej_reference_vocab_ps,
                                        **data_dict)
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size
        del diffusion_image, concatenated_images

        if self.args.dpo_token_weighted:
            uncond_ref_win_logp = self.compute_weighted_logp(uncond_ref_win_logp, win_labels, win_token_weight, self.args.dpo_use_average)
            uncond_ref_rej_logp = self.compute_weighted_logp(uncond_ref_rej_logp, rej_labels, rej_token_weight, self.args.dpo_use_average)
            ref_win_logp = self.compute_weighted_logp(ref_win_per_token_logp, win_labels, win_token_weight, self.args.dpo_use_average)
            ref_rej_logp = self.compute_weighted_logp(ref_rej_per_token_logp, rej_labels, rej_token_weight, self.args.dpo_use_average)
            concatenated_logp = self.compute_weighted_logp(concatenated_logp, concatenated_labels,concatenated_token_weight, self.args.dpo_use_average)

            if torch.any(torch.isnan(uncond_ref_win_logp)):
                print(f'uncond_ref_win_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(uncond_ref_rej_logp)):
                print(f'uncond_ref_rej_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(ref_win_logp)):
                print(f'ref_win_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(ref_rej_logp)):
                print(f'ref_rej_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(concatenated_logp)):
                print(f'concatenated_logp fail', flush=True)
                exit()


        policy_win_logp, policy_rej_logp, policy_win_diffusionImage_logps = concatenated_logp.split([win_size, rej_size, win_size])
        if self.args.use_tok:
            chosen_position_kl, rejected_position_kl = all_position_kl.split([win_size, rej_size])
        else:
            chosen_position_kl = rejected_position_kl = None

        if self.args.past_index >= 0:
            raise NotImplementedError
        
        losses, chosen_rewards, rejected_rewards = self.chip_loss(
                policy_win_logp, policy_rej_logp, policy_win_diffusionImage_logps,
                uncond_ref_win_logp, uncond_ref_rej_logp, 
                ref_win_logp, ref_rej_logp, 
                chosen_position_kl, rejected_position_kl
            )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # do SFT
        # loss = - policy_win_logp.mean()
        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()
        # loss = DPO_weight * losses.mean() - SFT_weight * policy_rej_logp.mean()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(policy_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(policy_win_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_rejected'] = self._nested_gather(ref_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_chosen'] = self._nested_gather(ref_win_logp.mean()).mean().item()
        # metrics[f'batch_size'] = len(win_labels)
        self.log(metrics)

        return loss

    def chip_loss(self, policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                policy_win_diffusionImage_logps: torch.FloatTensor,
                uncond_ref_win_logp: torch.FloatTensor,
                uncond_ref_rej_logp: torch.FloatTensor,
                reference_chosen_logps: torch.FloatTensor,
                reference_rejected_logps: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                reference_free: bool = False,
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            uncond_ref_win_logp: unconditional Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            uncond_ref_rej_logp: unconditional Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps 
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        if self.args.use_cross_modal_loss:
            logits += policy_chosen_logps - reference_chosen_logps
            logits -= policy_win_diffusionImage_logps - uncond_ref_win_logp
        if self.args.use_tok:
            logits -= self.args.tok_beta * (rejected_position_kl - chosen_position_kl.detach())
            chosen_rewards = self.args.dpo_beta * (policy_chosen_logps - reference_chosen_logps + chosen_position_kl).detach()
            rejected_rewards = self.args.dpo_beta * (policy_rejected_logps - reference_rejected_logps + rejected_position_kl).detach()
        else:
            chosen_rewards = self.args.dpo_beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.args.dpo_beta * (policy_rejected_logps - reference_rejected_logps).detach()
            
        losses = -F.logsigmoid(self.args.dpo_beta * logits)

        return losses, chosen_rewards, rejected_rewards


    def forward_DPO(self, model, input_ids, labels, attention_mask, images, win_reference_vocab_ps, rej_reference_vocab_ps, **kwargs):
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images,
            **kwargs
        )
        if self.args.use_tok:
            with open(f'{self.args.tok_logits_path}/{win_reference_vocab_ps[0]}.json', 'rb') as f:
                ref_logits = torch.tensor(json.load(f)).cuda()    
            _, all_position_kl, _ = tdpo_get_batch_logps(output.logits, ref_logits, labels)
        else:
            all_position_kl = None

        if self.args.dpo_token_weighted:
            token_log_prob = get_batch_logps(output.logits, labels, return_per_token_logp=True)
            return token_log_prob, all_position_kl
        else:
            log_prob, average_log_prob = get_batch_logps(output.logits, labels, return_per_token_logp=False)
            if self.args.dpo_use_average:
                return average_log_prob, all_position_kl

            return log_prob, all_position_kl

    @staticmethod
    def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
        loss_mask = (labels[:, 1:].clone() != -100)
        weighted_mask = token_weight * loss_mask
        logp = (per_token_logp * weighted_mask).sum(-1)

        average_logp = logp / weighted_mask.sum(-1)
        if use_average:
            return average_logp
        
        return logp

