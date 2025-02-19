import io
import os
import glob
import json
import base64
import random
import pathlib

from PIL import Image
from typing import List

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(keys, value):
            if not callable(value):
                raise Exception(
                    f"Register object must be callable! But receice:{value} is not callable!")

            if not isinstance(keys, list):
                keys = [keys]

            for key in keys:
                if key in self._dict:
                    print(
                        f"error: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
                    exit()

                self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


register_data_processor = Register()
register_data_path = Register()


def vqa_instruction_templates(question, idx=None):
    instructions = [
        "{Question} A short answer to the question is",
        "Given the image, answer the following question with no more than three words. {Question}",
        "Based on the image, respond to this question with a short answer: {Question} Answer:",
        "Use the provided image to answer the question: {Question} Provide your answer as short as possible:",
    ]
    if idx is None:
        new_question = random.choice(instructions).replace("{Question}", question)
    else:
        new_question = instructions[idx].replace("{Question}", question)

    return new_question


def caption_instruction_templates():
    instructions = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

    new_question = random.choice(instructions)

    return new_question


def load_multimodal_conversation(text_b64, img_b64_buffer):
    map_role = {
        'human': 'human',
        'gpt': 'gpt'
    }

    text = base64.b64decode(text_b64).decode('utf-8')
    list_conv = json.loads(text)

    out: List[dict] = []
    for idx, sentence in enumerate(list_conv):
        value = sentence['value']

        if idx == 0 and '<image>' not in value:
            value = f"<image>\n{value}"
        if idx != 0 and '<image>' in value:
            value = value.replace('<image>', '')

        out.append({
            'from': map_role[sentence['from']],
            'value': value
        })

    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image, out


def b64_to_PIL_image(img_b64_buffer):
    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image


def wrap_qa_to_single_turn_multimodal_conv(answer, question):
    if '<image>' not in question:
        question = f"<image>\n{question}"

    out = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer}
    ]
    return question, out


def wrap_generation_single_turn_conv(out, template_func):
    conv = [
        {
            "from": "human",
            "value": f"<image>\n{template_func()}"

        },
        {
            "from": "gpt",
            "value": out
        }
    ]
    return conv


def wrap_caption_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, caption_instruction_templates)


def gather_data_files_by_glob(root: str, pattern='*.tsv'):
    filenames = []

    for fullpath in glob.glob(f'{root}/{pattern}'):
        filename = fullpath.split('/')[-1]
        filenames.append(filename)
    return root, filenames


@register_data_path('unimm-chat')
def unimmchat_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../data/unimm-chat'
    return gather_data_files_by_glob(data_dir, '*.tsv')


@register_data_processor(['unimm-chat'])
def unimmchat_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented


@register_data_processor('playground/data')
def dpo_cvpr_ncrp_vqa_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)


def dpo_preference_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        question = list_conv[0]
        if '<image>' not in question:
            question = f"<image>\n{question}"

        out_chosen = list_conv[1]
        out_rejected = list_conv[2]

        question = {"from": "human", "value": question}
        out_chosen = {"from": "gpt", "value": out_chosen}
        out_rejected = {"from": "gpt", "value": out_rejected}

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        data_dict = {
            'image': image,
            'question': question,
            'chosen': out_chosen,
            'rejected': out_rejected,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        if len(list_conv) == 4:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
             data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = list_conv[3]

        return data_dict
    else:
        raise NotImplemented


@register_data_path('vqav2-val')
def vqav2_val_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../data/VQAv2'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'val' in f]
    return data_dir, filenames


@register_data_processor('vqav2-val')
def vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": int(origin_split_inner_idx),
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa["question"],
        }
    else:
        raise NotImplemented


@register_data_path('vqav2-train')
def vqav2_train_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../data/VQAv2'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'train' in f]
    return data_dir, filenames


@register_data_processor('vqav2-train')
def vqav2_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]
        question = vqa_instruction_templates(question)  # vqa short answer template

        question, out = wrap_qa_to_single_turn_multimodal_conv(answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    elif intent == 'eval':
        return vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                   intent, img_transformer)
    else:
        raise NotImplemented
