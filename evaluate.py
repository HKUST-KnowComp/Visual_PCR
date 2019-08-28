#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import util
import numpy as np
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='evaluate pronoun resolution on trained model')
parser.add_argument('model', type=str,
                    help='model name to evaluate')
parser.add_argument('--split', type=str, default='test',
                    help='split to evaluate, test or val')
parser.add_argument('--output_dir', type=str, default='output',
                    help='output dir')

pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']


def main(args):
    # load data
    evaluate_file = f'{args.split}.vispro.1.1.prediction.jsonlines'
    evaluate_file = osp.join(args.output_dir, args.model, evaluate_file)
    test_data = list()
    with open(evaluate_file, 'r') as f:
        for line in f:
            tmp_example = json.loads(line)
            test_data.append(tmp_example)
    print(f'Evaluate prediction of {evaluate_file}')

    # initialize variables
    eval_types = ['all', 'not_discussed', 'discussed']
    all_antecedent = {key: 0 for key in eval_types}
    predict_antecedent = {key: 0 for key in eval_types}
    correct_antecedent = {key: 0 for key in eval_types}

    # deal with each dialog
    for i, tmp_example in enumerate(test_data):
        all_sentence = list()
        caption_len = len(tmp_example['sentences'][0])
        predicted_clusters = [tuple(pc) for pc in tmp_example['predicted_clusters']]

        for s in tmp_example['sentences']:
            all_sentence += s

        tokens_cdd = tmp_example['cdd_sentences']
        cdd_tokens_len = [len(t) for t in tokens_cdd]
        cdd_tokens_end = np.cumsum(cdd_tokens_len) + len(all_sentence)
        cdd_nps = []
        for cdd_end, cdd_len in zip(cdd_tokens_end, cdd_tokens_len):
            cdd_nps.append([cdd_end - cdd_len, cdd_end - 1])

        for pronoun_example in tmp_example['pronoun_info']:
            tmp_pronoun = all_sentence[pronoun_example['current_pronoun'][0]]
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]

            tmp_candidate_NPs = pronoun_example['candidate_NPs']
            tmp_candidate_NPs += cdd_nps
            tmp_candidate_NPs = tuple(tmp_candidate_NPs)
            tmp_correct_candidate_NPs = tuple(pronoun_example['correct_NPs'])
            not_discussed_flag = pronoun_example['not_discussed']
            if not_discussed_flag:
                all_antecedent['not_discussed'] += len(tmp_correct_candidate_NPs)
            else:
                all_antecedent['discussed'] += len(tmp_correct_candidate_NPs)
            all_antecedent['all'] += len(tmp_correct_candidate_NPs)             

            find_pronoun = False
            for cluster_id, coref_cluster in enumerate(predicted_clusters):
                for mention in coref_cluster:
                    if mention[0] == tmp_pronoun_index:
                        find_pronoun = True
                        find_cluster_id = cluster_id
                if find_pronoun:
                    break
            if find_pronoun and pronoun_example['reference_type'] == 0:
                coref_cluster = predicted_clusters[find_cluster_id]
                matched_cdd_np_ids = []
                matched_crr_np_ids = []
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    tmp_mention_span = (
                        mention_start_index,
                        mention[1])
                    matched_np_id = util.verify_correct_NP_match(tmp_mention_span, tmp_candidate_NPs, 'cover', matched_cdd_np_ids)
                    if matched_np_id is not None:
                        # exclude such scenario: predict 'its' and overlap with candidate 'its eyes'
                        # predict +1 but correct +0
                        if tmp_mention_span[0] < len(all_sentence) and\
                            tmp_mention_span[0] == tmp_mention_span[1] and\
                            all_sentence[tmp_mention_span[0]] in pronoun_list and\
                            len(tmp_candidate_NPs[matched_np_id]) > 1:
                            continue
                        matched_cdd_np_ids.append(matched_np_id)
                        predict_antecedent['all'] += 1
                        if not_discussed_flag:
                            predict_antecedent['not_discussed'] += 1
                        else:
                            predict_antecedent['discussed'] += 1
                        matched_np_id = util.verify_correct_NP_match(tmp_mention_span, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                        if matched_np_id is not None:
                            matched_crr_np_ids.append(matched_np_id)
                            correct_antecedent['all'] += 1
                            if not_discussed_flag:
                                correct_antecedent['not_discussed'] += 1
                            else:
                                correct_antecedent['discussed'] += 1

    print('Pronoun resolution')
    results = []
    for key in ['discussed', 'not_discussed', 'all']:
        p = 0 if predict_antecedent[key] == 0 else correct_antecedent[key] / predict_antecedent[key]
        r = 0 if all_antecedent[key] == 0 else correct_antecedent[key] / all_antecedent[key]
        f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
        results.extend([p, r, f1])
        print(key)
        print(f'\tP: {p * 100:.2f}, R: {r * 100:.2f}, F1: {f1 * 100:.2f}')
        print(f'\tall: {all_antecedent[key]}, predict: {predict_antecedent[key]}, correct: {correct_antecedent[key]}')

    return results
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)    
