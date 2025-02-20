import io
import os
import json
import tqdm
import copy
import random
import torch
import itertools
import pandas as pd
import torch.utils.data as torch_data
import PIL.Image as PIL_image
import torchvision.transforms as transforms

from functools import partial

from muffin.train.train_utils import encode_multimodal_preference_sample, SFT_collator_fn


def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = PIL_image.open(img_io).convert('RGB')
    return image


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                          average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    labels = labels[:-1, :].clone()
    logits = logits[:-1, :, :]

    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob, logits

    return log_prob, average_log_prob


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 image_token_len,
                 img_processor,
                 use_im_start_end=True):

        self.data = data

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": json.loads(sample['origin_split']),
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        text = json.loads(sample['text'])
        question = {'from': 'human', 'value': f"<image>\n{text['question']}"}
        chosen = {'from': 'gpt', 'value': text['chosen']}
        rejected = {'from': 'gpt', 'value': text['rejected']}
        
        if 'bytes' in sample['image']:
            image = bytes_to_PIL_image(sample['image']['bytes'])
        else:
            with open(sample['image']['path'], 'rb') as f:
                image = bytes_to_PIL_image(f.read())

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        # rlhf-v data random_image
        if 'bytes' in sample['image']:
            formated_sample['random_image'] = bytes_to_PIL_image(self.data[random.choice(range(len(self.data)))]['image']['bytes'])
            
            height, width = image.size
            resize_height = min(height, 240)
            resize_width = min(width, 320)
            # RandomCrop
            RandomCrop = transforms.RandomCrop(size=(resize_width, resize_height)) 
            formated_sample['crop_image'] = RandomCrop(image)
            
            RR = transforms.RandomRotation(degrees=(10, 80))  
            formated_sample['rotate_image'] = RR(image)
            
        # other image
        if 'random_image_path' in sample:
            formated_sample['random_image'] = bytes_to_PIL_image(open(sample['random_image_path'], 'rb').read())
        if 'crop_image_path' in sample:
            formated_sample['crop_image'] = bytes_to_PIL_image(open(sample['crop_image_path'], 'rb').read())
        if 'rotate_image_path' in sample:
            formated_sample['rotate_image'] = bytes_to_PIL_image(open(sample['rotate_image_path'], 'rb').read())
            

        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(formated_sample, self.tokenizer, self.mm_cfg)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)


def pretty_print(data_dict, tokenizer):
    input_ids = data_dict['input_ids']
    input_str = tokenizer.decode(input_ids)
    print(f'input_ids.shape={input_ids.shape}\ninput_str is {input_str}')

    label_ids = data_dict['labels']
    print(f'label_ids.shape={input_ids.shape}')
    for i, o in zip(input_ids, label_ids):
        i_tok = tokenizer.convert_ids_to_tokens(i.item())
        o_tok = tokenizer.convert_ids_to_tokens(o.item()) if o.item() != -100 else '[SKIP]'
        print(f'{i_tok:10s} => {o_tok:10s}')


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_three(tensorA, tensorB, tensorC, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC),
        batch_first=True,
        padding_value=padding_value)
    return out


def preference_collator_fn(instances, pad_token_id, mdpo=False):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)
    
    if mdpo is True: 
        concatenated_input_ids = concate_pad_three(win_batch['input_ids'], rej_batch['input_ids'], win_batch['input_ids'], pad_token_id)
        concatenated_labels = concate_pad_three(win_batch['labels'], rej_batch['labels'], win_batch['labels'], -100)
        concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)
    else:
        concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
        concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
        concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)


    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'],
        diffusion_image=win_batch.get('diffusion_image'),
        random_image=win_batch.get('random_image'),
        crop_image=win_batch.get('crop_image'),
        rotate_image=win_batch.get('rotate_image'),
        idx=rej_instances[0]['idx']
    )
    return batch


def get_multimodal_sample_logps(model, dataloader, pad_token_id=None, 
                                tok_logits_path='', use_tok=False, 
                                use_image_type='diffusion'):
    uncond_win_logp_list = []
    uncond_rej_logp_list = []
    win_reference_vocab_ps = []
    rej_reference_vocab_ps = []

    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []
    
    rank_num = torch.distributed.get_rank()
    line_idx = 0

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):
            line_idx += 1
            if use_tok is True:
                if not os.path.exists(tok_logits_path):
                    os.makedirs(tok_logits_path)
                    concatenated_input_ids = concate_pad(batch['win_input_ids'], batch['rej_input_ids'], pad_token_id).cuda()
                    concatenated_labels = concate_pad(batch['win_labels'], batch['rej_labels'], -100).cuda()
                    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id).cuda()
                    concatenated_images = torch.cat([batch['images'].half().cuda(), batch['images'].half().cuda()], dim=0)
                    output = model(
                        input_ids=concatenated_input_ids,
                        labels=concatenated_labels,
                        attention_mask=concatenated_attention_mask,
                        images=concatenated_images
                    )
                    ref_logits = output.logits
                    with open(f'{tok_logits_path}/{batch["idx"]}.json', 'w', encoding='utf-8') as wf:
                        wf.write(json.dumps(ref_logits.tolist()))


            for key in ['win', 'rej']:
                input_ids = batch[f'{key}_input_ids'].cuda()
                labels = batch[f'{key}_labels'].cuda()
                attention_mask = batch[f'{key}_attention_mask'].cuda()

                if use_image_type == 'diffusion':
                    tmp_image = batch['diffusion_image'].half().cuda()
                elif use_image_type == 'black':
                    tmp_image = torch.zeros_like(batch['images'].half().cuda())
                elif use_image_type == 'crop':
                    tmp_image = batch['crop_image'].half().cuda()
                elif use_image_type == 'rotate':
                    tmp_image = batch['rotate_image'].half().cuda()
                elif use_image_type == 'random':
                    tmp_image = batch['random_image'].half().cuda()

                uncond_output = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    images=tmp_image
                )
                uncond_log_prob, _ = get_batch_logps(uncond_output.logits, labels)
                uncond_log_prob = uncond_log_prob.tolist()

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    images=batch['images'].half().cuda()
                )
                per_token_logp, log_prob, average_log_prob, ref_logits = get_batch_logps(output.logits, labels, return_all=True)

                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                log_prob = log_prob.tolist()
                average_log_prob = average_log_prob.tolist()

                if key == 'win':
                    uncond_win_logp_list += uncond_log_prob
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                    win_reference_vocab_ps += [batch["idx"]]
                else:
                    uncond_rej_logp_list += uncond_log_prob
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp
                    rej_reference_vocab_ps += [batch["idx"]]

    return uncond_win_logp_list, win_logp_list, win_avg_logp_list, win_per_token_logp_list, uncond_rej_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, win_reference_vocab_ps, rej_reference_vocab_ps


def write_logp_to_preference_parquet(origin_data, cache_file, logps, overwrite_logps=False):
    out_data = []

    for index in range(len(origin_data)):
        line = origin_data[index]
        logp_data = logps[index]

        new_line = copy.deepcopy(line)

        text = json.loads(new_line['text'])

        if 'logps' in text.keys():
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            text['logps'] = logp_data
            new_line['text'] = json.dumps(text)

        else:
            assert list(text.keys()) == ['question', 'chosen', 'rejected'], f'Undefined data structure, expecting [Q, Win, Rej], got {text.keys()}'
            text['logps'] = logp_data
            new_line['text'] = json.dumps(text)

        out_data.append(new_line)

    df = pd.DataFrame(out_data)

    if torch.distributed.get_rank() == 0:
        df.to_parquet(cache_file)

    torch.distributed.barrier()

    return df

def inference_logp(model, tokenizer, hf_data, cache_file, 
                   image_token_len, img_processor, use_im_start_end,
                   tok_logits_path, use_tok, use_image_type
                   ):
    model = model.to(dtype=torch.bfloat16, device='cuda')
    dataset = PreferenceInferenceDataset(tokenizer=tokenizer,
                                    data = hf_data,
                                    image_token_len=image_token_len,
                                    img_processor=img_processor,
                                    use_im_start_end=use_im_start_end)
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False, sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model, dataloader, 
                                          pad_token_id=tokenizer.pad_token_id,
                                          tok_logits_path=tok_logits_path,
                                          use_tok=use_tok,
                                          use_image_type=use_image_type)

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]

    uncond_win_logp_list, win_logp_list, win_avg_logp_list, win_per_token_logp_list, uncond_rej_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, \
    win_reference_vocab_ps, rej_reference_vocab_ps = merged_outputs

    logps = list(zip(uncond_win_logp_list, win_logp_list, win_avg_logp_list, win_per_token_logp_list, uncond_rej_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, win_reference_vocab_ps, rej_reference_vocab_ps))

    df = write_logp_to_preference_parquet(dataset.data, cache_file, logps, overwrite_logps=False)

    torch.distributed.barrier()

    return df