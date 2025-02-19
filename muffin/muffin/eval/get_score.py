import os
import json
import argparse
import time
import pandas as pd
import re
import sys
import concurrent.futures
from nltk.tokenize import word_tokenize
import editdistance
import tiktoken
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from cider import CiderScorer

print(sys.setrecursionlimit(10 ** 5))

enc = tiktoken.get_encoding("cl100k_base")


def num_list_edit_distance(a, b):
    return editdistance.eval(enc.encode(a), enc.encode(b))


def string_list_edit_distance(token_reference, token_candidate):
    return editdistance.eval(token_reference, token_candidate)


def string_edit_distance(a, b):
    return editdistance.eval(a, b)


def meteor(token_reference, token_candidate):
    return meteor_score([token_reference], token_candidate)


def rouge_func(candidate, reference):
    if not candidate:
        return 0
    rouge = Rouge()
    return rouge.get_scores(candidate, reference, avg=True)['rouge-l']['f']


def bleu_func_1(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[1, 0, 0, 0])


def bleu_func_2(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0, 1, 0, 0])


def bleu_func_3(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0, 0, 1, 0])


def bleu_func_4(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0, 0, 0, 1])


def cum_bleu_func_1(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[1, 0, 0, 0])


def cum_bleu_func_2(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0.5, 0.5, 0, 0])


def cum_bleu_func_3(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0.33, 0.33, 0.33, 0])


def cum_bleu_func_4(token_reference, token_candidate):
    return sentence_bleu([token_reference], token_candidate, weights=[0.25, 0.25, 0.25, 0.25])


def one_file(file):
    sample_count = 0

    chosen_sl_ed_sum = 0
    chosen_bleu_1_sum = 0
    chosen_bleu_4_sum = 0
    cum_chosen_bleu_4_sum = 0
    meteor_sum = 0
    rouge_sum = 0
    # cider
    cider_scorer = CiderScorer()
    file_result_list = []
    with open(file, 'rb') as rf:
        for line in rf:
            try:
                item = json.loads(line)
            except Exception as e:
                print(e)
                continue
            item['response'] = re.sub(r'^Assistant:', '', item['response']).strip()

            token_reference = word_tokenize(item['chosen'])
            token_candidate = word_tokenize(item['response'])
            chosen_sl_ed_sum += string_list_edit_distance(token_reference, token_candidate)
            _chosen_bleu_1 = bleu_func_1(token_reference, token_candidate)
            item['bleu_1'] = _chosen_bleu_1
            chosen_bleu_1_sum += _chosen_bleu_1
            _chosen_bleu_4 = bleu_func_4(token_reference, token_candidate)
            item['bleu_4'] = _chosen_bleu_4
            chosen_bleu_4_sum += _chosen_bleu_4
            cum_chosen_bleu_4_sum += cum_bleu_func_4(token_reference, token_candidate)
            meteor_score = meteor(token_reference, token_candidate)
            item['meteor'] = meteor_score
            meteor_sum += meteor_score
            try:
                rouge_score = rouge_func(item['response'], item['chosen'])
                item['rouge'] = rouge_score
                rouge_sum += rouge_score
            except Exception as e:
                print(file, 'resp:', item['response'], 'chosen:', item['chosen'], '???', e)

            sample_count += 1
            file_result_list.append(item)

            cider_scorer += (item['response'], [item['chosen']])
    _score_file_l, _score_file_r = file.split('.', maxsplit=1)
    score_file = _score_file_l + '_score.' + _score_file_r
    with open(score_file, 'w', encoding='utf-8') as f:
        file_result_list.sort(key=lambda x: x['idx'])
        f.write(json.dumps(file_result_list, ensure_ascii=False, indent=2))

    if cider_scorer.crefs:
        (cider_score, _) = cider_scorer.compute_score()
    else:
        return

    print('file: ', file)
    print('samples count: ', sample_count)
    print('*' * 10)
    if sample_count == 0:
        return
    file_name = file.split('/')[-1]

    tmp_dict = {
        'file': file_name,
        'bleu_4': round(chosen_bleu_4_sum / sample_count, 5) * 100,
        'METEOR': round(meteor_sum / sample_count, 5) * 100,
        'ROUGE-L': round(rouge_sum / sample_count, 5) * 100,
        'CIDEr': round(cider_score, 5) * 100,
        'bleu_1': round(chosen_bleu_1_sum / sample_count, 5) * 100,
        'cumulative_bleu_4': round(cum_chosen_bleu_4_sum / sample_count, 5) * 100,
        'edit_distance': round(chosen_sl_ed_sum / sample_count, 3),
        'sample_count': sample_count,
    }
    return tmp_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    file = args.file
    result_list = []

    file_base = 'eval_output/'

    file_list = os.listdir(file_base)
    futures = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        for _file in file_list:
            futures[executor.submit(one_file, os.path.join(file_base, _file))] = _file
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            if result:
                result_list.append(result)
        except Exception as e:
            raise (e)
    result_list.sort(key=lambda x: len(x['file']), reverse=False)

    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join('./eval_result', 'eval_result.csv'))





