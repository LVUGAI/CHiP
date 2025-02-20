import os
import json
import argparse
import pandas as pd


def eval_amber():
    file_list = [
        'bs3_amber_muffinRVS.json'
    ]

    for concat_file in file_list:
        result_list = []
        if '_concat' in concat_file:
            prefix = concat_file.split('_concat')[0]
            qid_set = set()
            with open(os.path.join(base_path, concat_file), 'w', encoding='utf-8') as wf:
                for _file in os.listdir(base_path):
                    if _file.startswith(prefix) and '_concat' not in _file:
                        if all(x.isdigit() for x in _file.replace(prefix, '').replace('.json', '').split('_') if x):
                            with open(os.path.join(base_path, _file), 'rb') as f:
                                for x in f:
                                    tmp_data = json.loads(x)
                                    qid = tmp_data['chosen']
                                    if qid in qid_set:
                                        continue
                                    qid_set.add(qid)
                                    response = tmp_data['response'].replace('Assistant:', '').strip()
                                    if qid > 1004:
                                        if response.lower().startswith('no'):
                                            response = 'No'
                                        elif response.lower().startswith('yes'):
                                            response = 'Yes'
                                        else:
                                            response = ''
                                    result_list.append({'id':qid, 'response': response})
                result_list.sort(key=lambda x: x['id'])
                wf.write(json.dumps(result_list, ensure_ascii=False, indent=2))
            print(len(qid_set), len(result_list), concat_file)
        else:
            with open(os.path.join(base_path, 'amber_' + concat_file), 'w', encoding='utf-8') as wf:
                with open(os.path.join(base_path, concat_file), 'rb') as f:
                    for x in f:
                        tmp_data = json.loads(x)
                        qid = tmp_data['chosen']
                        response = tmp_data['response'].replace('Assistant:', '').strip()
                        if qid > 1004:
                            if response.lower().startswith('no'):
                                response = 'No'
                            elif response.lower().startswith('yes'):
                                response = 'Yes'
                            else:
                                response = ''
                        result_list.append({'id':qid, 'response': response})                
                wf.write(json.dumps(result_list, ensure_ascii=False, indent=2))                        
            print(len(result_list))
    

def eval_hallusionbench():
    raw_data_map = dict()
    with open('/project/HallusionBench/HallusionBench.json', 'rb') as f:
        for idx, data in enumerate(json.load(f)):
            data['idx'] = idx
            raw_data_map[idx] = data

    file_list = [
        "bs3_hallusionbech_muffinRVS.json",
    ]
    for concat_file in file_list:
        result_list = []
        if '_concat' in concat_file:
            prefix = concat_file.split('_concat')[0]
            qid_set = set()
            with open(os.path.join(base_path, concat_file), 'w', encoding='utf-8') as wf:
                for _file in os.listdir(base_path):
                    if _file.startswith(prefix) and '_concat' not in _file:
                        if all(x.isdigit() for x in _file.replace(prefix, '').replace('.json', '').split('_') if x):
                            with open(os.path.join(base_path, _file), 'rb') as f:
                                for x in f:
                                    tmp_data = json.loads(x)
                                    qid = tmp_data['chosen']
                                    if qid in qid_set:
                                        continue
                                    qid_set.add(qid)
                                    response = tmp_data['response'].replace('Assistant:', '').strip()
                                    raw_data_map[qid]['model_prediction'] = response
                                    result_list.append(raw_data_map[qid])
                result_list.sort(key=lambda x: x['idx'])
                wf.write(json.dumps(result_list, ensure_ascii=False, indent=2))
            print(len(qid_set), len(result_list), concat_file)    
        else:
            with open(os.path.join(base_path, 'hallusionbench_' + concat_file), 'w', encoding='utf-8') as wf:
                with open(os.path.join(base_path, concat_file), 'rb') as f:
                    for x in f:
                        tmp_data = json.loads(x)
                        qid = tmp_data['chosen']
                        response = tmp_data['response'].replace('Assistant:', '').strip()
                        raw_data_map[qid]['model_prediction'] = response
                        result_list.append(raw_data_map[qid])
                result_list.sort(key=lambda x: x['idx'])
                wf.write(json.dumps(result_list, ensure_ascii=False, indent=2))
            print(len(result_list), concat_file)    
                




if __name__ == "__main__":
    base_path='./eval_output/'
    eval_amber()
    # eval_hallusionbench()

