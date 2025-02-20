import os
import gc
import copy
import time
import transformers
from PIL import Image
import json
import torch

from typing import Dict, Optional, Sequence
from muffin import conversation as conversation_lib

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def SFT_collator_fn(instances, pad_token_id):
    input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(pad_token_id),
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
    # other image
    if 'diffusion_image' in instances[0]:
        batch['diffusion_image'] = torch.stack([instance['diffusion_image'] for instance in instances])
    if 'random_image' in instances[0]:
        batch['random_image'] = torch.stack([instance['random_image'] for instance in instances])
    if 'crop_image' in instances[0]:
        batch['crop_image'] = torch.stack([instance['crop_image'] for instance in instances])
    if 'rotate_image' in instances[0]:
        batch['rotate_image'] = torch.stack([instance['rotate_image'] for instance in instances])
    
    return batch


def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg):
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    if 'image' in source:
        image =  source['image']
        image = multimodal_cfg['image_processor'](image)
        win_conv = expand_image_token(win_conv, multimodal_cfg)
        rej_conv = expand_image_token(rej_conv, multimodal_cfg)

    rej_data_dict = preprocess([rej_conv], tokenizer)
    rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                         labels=rej_data_dict["labels"][0])

    win_data_dict = preprocess([win_conv], tokenizer)
    win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                         labels=win_data_dict["labels"][0])
    # other image
    if 'random_image' in source:
        rej_data_dict['random_image'] = win_data_dict['random_image'] = multimodal_cfg['image_processor'](source['random_image'])
    if 'crop_image' in source:
        rej_data_dict['crop_image'] = win_data_dict['crop_image'] = multimodal_cfg['image_processor'](source['crop_image'])
    if 'rotate_image' in source:
        rej_data_dict['rotate_image'] = win_data_dict['rotate_image'] = multimodal_cfg['image_processor'](source['rotate_image'])

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image
        rej_data_dict['diffusion_image'] = win_data_dict['diffusion_image'] = add_diffusion_noise(image, 500)
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

    if 'ref_win_logp' in source:
        rej_data_dict['uncond_ref_rej_logp'] = source['uncond_ref_rej_logp']
        win_data_dict['uncond_ref_win_logp'] = source['uncond_ref_win_logp']
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
        rej_data_dict['rej_reference_vocab_ps'] = source['rej_reference_vocab_ps']
        win_data_dict['win_reference_vocab_ps'] = source['win_reference_vocab_ps']
    rej_data_dict['idx'] = win_data_dict['idx'] = source['idx']
    return rej_data_dict, win_data_dict

def expand_image_token(source, multimodal_cfg) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = multimodal_cfg['image_token_len']
    if not is_multimodal:
        return source

    for sentence in source:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        if multimodal_cfg['use_im_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return source

def encode_multimodal_sample(source, tokenizer, multimodal_cfg):
    conversation = copy.deepcopy(source["conversations"])
    if 'image' in source:
        image =  source['image']
        image = multimodal_cfg['image_processor'](image)
        conversation = expand_image_token(conversation, multimodal_cfg)

    data_dict = preprocess([conversation], tokenizer)
    data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
    return data_dict

def encode_alttext_multimodal_sample(source, tokenizer, multimodal_cfg):
    text = json.loads(source['text'])
    if source['ds_name'] == "":
        prompt = 'Given an image and context, generate alt-text for the image.\nContext:\n'
        conversation = [
            # {'from': 'human', 'value': f"<image>\n{text['question']}"},
            {'from': 'human', 'value': f"<image>\n{prompt}{text['rejected']}"},
            {'from': 'gpt', 'value': f"{text['chosen']}"}
            ]
    elif source['ds_name'] == "":
        conversation = [
            {'from': 'human', 'value': f"<image>\n{text['question']}"},
            {'from': 'gpt', 'value': f"{text['chosen']}"}
            ]
    image = Image.open(source['image_path']).convert('RGB')
    image = multimodal_cfg['image_processor'](image)
    conversation = expand_image_token(conversation, multimodal_cfg)

    data_dict = preprocess([conversation], tokenizer)
    data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
    return data_dict


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
