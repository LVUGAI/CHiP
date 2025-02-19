import json
import argparse
import torch
import time
import os
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import base64
from PIL import Image
import datasets
import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def each_task(args, row, tokenizer, model, image_processor, print_one=False):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + row['question'] + '\n'
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if print_one is True:
        print('>>>>>> prompt:', prompt)

    if 'image_path' in row:
        images = load_images([row['image_path']])
    else:
        images = [Image.open(row.pop('image_bytes')).convert("RGB")]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            repetition_penalty=1.0,
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def all_task(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_name, None, model_name
    )

    if args.test_datasets == 'obj_halbench_300':
        with open('playground/data/obj_halbench_300_with_image.jsonl', 'r') as f:
            data_list = []
            for x in f:
                data = json.loads(x)
                data['image_bytes'] = BytesIO(base64.b64decode(data['image']))
                data.pop('image_path', '')
                data_list.append(data)
    elif args.test_datasets == 'mmhal_bench':
        img_path = 'playground/data/mmhal-bench_with_image.jsonl'
        with open(img_path, 'r') as f:
            data_list = []
            for x in f:
                data = json.loads(x)
                data['image_bytes'] = BytesIO(base64.b64decode(data['image']))
                data.pop('image_path', '')
                data_list.append(data)
    elif args.test_datasets == 'amber':
        data_list = []
        img_path = 'playground/data/amber_image/'
        with open('playground/data/query_all.json', 'rb') as f:
            for data in json.load(f):
                data['question'] = data['query']
                if data['id'] > 1004:
                    data['question'] += '\nAnswer the question using a single word: Yes or No.\n'
                    continue
                data['image_path'] = img_path + data['image']
                data_list.append(data)
    elif args.test_datasets == 'HallusionBench':
        data_list = []
        with open('playground/data/HallusionBench.json', 'rb') as f:
            for idx, data in enumerate(json.load(f)):
                data['idx'] = idx
                if data['filename']:
                    data['image'] = 'playground/data/HallusionBench_image/' + data['filename'][2:]
                else:
                    data['image_path'] = None
                data['question'] += '\nAnswer the question using a single word: Yes or No.\n'                    
                data_list.append(data)
    else:
        with open(args.test_datasets, 'rb') as f:
            data_list = [json.loads(x) for x in f]
    eval_output = args.eval_output

    if args.begin!=-1:
        part = len(data_list)//args.total
        _begin = args.begin*part
        _end = (args.begin+1)*part if args.begin!=(args.total-1) else len(data_list)
        data_list = data_list[_begin:_end]
        eval_output = eval_output.split('.')[0]+f'_{_begin}_{_end}.json'
        print('>>> length: ', len(data_list), 'begin: ', _begin, 'end: ', _end)

    add_prefix = f'bs{args.num_beams}_'
    eval_output = os.path.join(os.path.dirname(eval_output), add_prefix + eval_output.split('/')[-1])

    with open(eval_output, "w", encoding='utf-8') as ans_file:
        print_one = True
        for row in tqdm(data_list):
            row.pop('image', '')
            row['response'] = each_task(args, row, tokenizer, model, image_processor, print_one=print_one)
            ans_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            ans_file.flush()
            print_one = False
    print('file:', eval_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_datasets", type=str, 
        # default="obj_halbench_300"
        # default="mmhal_bench"
        # default="HallusionBench"
        default="amber"
    )

    parser.add_argument(
        "--model_name", type=str,
        default="liuhaotian/llava-v1.6-vicuna-7b"
    )

    parser.add_argument(
        "--eval_output", type=str,
        default="amber_llava_v16_vicuna_7b.json"
    )

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--begin", type=int, default=-1)
    parser.add_argument("--total", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=-1)
    
    args = parser.parse_args()
    print(args)
    if args.sleep!=-1:
        print('sleep: ', args.sleep)
        time.sleep(args.sleep)

    all_task(args)
