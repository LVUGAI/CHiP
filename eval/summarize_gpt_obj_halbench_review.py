import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str,default="")
    args = parser.parse_args()
    review_file = args.cap_file
    data = json.load(open(review_file))
    metrics = data['overall_metrics']
    hall_response = metrics['CHAIRs_refine'] * 100
    obj_hall_rate = metrics['CHAIRi'] * 100
    correct_response = metrics['correct_rate'] * 100
    obj_correct_rate = metrics['object_correct_rate'] * 100
    obj_recall = metrics['obj_rec'] * 100
    coco_sentence_num = metrics['coco_sentence_num']
    coco_word_count = metrics['coco_word_count']
    gt_word_count = metrics['gt_word_count']
    avg_length = metrics['avg_word_len']

    obj_f1 = 2 * obj_recall * obj_correct_rate / (obj_recall + obj_correct_rate)
    res_f1 = 2 * (coco_sentence_num / 3) * correct_response / (coco_sentence_num / 3 + correct_response)

    print('file: ', review_file)
    print(f'Response Hall   : {hall_response:.2f}\n'
            f'Object Hall     : {obj_hall_rate:.2f}\n'
            f'Response Correct: {correct_response:.2f}\n'
            f'Object Correct  : {obj_correct_rate:.2f}\n'
            f'Object Recall   : {obj_recall:.2f}\n'
            f'Average Length  : {avg_length:.2f}\n'
            f'COCO Sent Number: {coco_sentence_num}\n'
            f'COCO Word Number: {coco_word_count}\n'
            f'GT Word Number  : {gt_word_count}')
