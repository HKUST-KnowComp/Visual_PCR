from __future__ import absolute_import
from __future__ import division

import util
import numpy as np


class PrCorefEvaluator(object):
    def __init__(self):
        self.all_coreference = 0
        self.predict_coreference = 0
        self.correct_predict_coreference = 0
        self.all_coref_discussed = 0
        self.predict_coref_discussed = 0
        self.correct_predict_coref_discussed = 0
        self.all_coref_not_discussed = 0
        self.predict_coref_not_discussed = 0
        self.correct_predict_coref_not_discussed = 0

        self.pronoun_list = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them', 'They', 'it', 'It', 'his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

    def get_prf(self):
        results = {}
        results['p'] = 0 if self.predict_coreference == 0 else self.correct_predict_coreference / self.predict_coreference
        results['r'] = 0 if self.all_coreference == 0 else self.correct_predict_coreference / self.all_coreference
        results['f'] = 0 if results['p'] + results['r'] == 0 else 2 * results['p'] * results['r'] / (results['p'] + results['r'])
        results['p_discussed'] = 0 if self.predict_coref_discussed == 0 else self.correct_predict_coref_discussed / self.predict_coref_discussed
        results['r_discussed'] = 0 if self.all_coref_discussed == 0 else self.correct_predict_coref_discussed / self.all_coref_discussed
        results['f_discussed'] = 0 if results['p_discussed'] + results['r_discussed'] == 0 else 2 * results['p_discussed'] * results['r_discussed'] / (results['p_discussed'] + results['r_discussed'])
        results['p_not_discussed'] = 0 if self.predict_coref_not_discussed == 0 else self.correct_predict_coref_not_discussed / self.predict_coref_not_discussed
        results['r_not_discussed'] = 0 if self.all_coref_not_discussed == 0 else self.correct_predict_coref_not_discussed / self.all_coref_not_discussed
        results['f_not_discussed'] = 0 if results['p_not_discussed'] + results['r_not_discussed'] == 0 else 2 * results['p_not_discussed'] * results['r_not_discussed'] / (results['p_not_discussed'] + results['r_not_discussed'])

        return results


    def update(self, predicted_clusters, pronoun_info, sentences, tokens_cdd):
        all_sentence = list()
        caption_len = len(sentences[0])
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]

        for s in sentences:
            all_sentence += s

        cdd_tokens_len = np.sum(tokens_cdd != '', axis=1)
        cdd_tokens_end = np.cumsum(cdd_tokens_len) + len(all_sentence)
        cdd_nps = []
        for cdd_end, cdd_len in zip(cdd_tokens_end, cdd_tokens_len):
            cdd_nps.append([cdd_end - cdd_len, cdd_end - 1])

        for pronoun_example in pronoun_info:
            tmp_pronoun_index = pronoun_example['current_pronoun'][0]

            tmp_candidate_NPs = pronoun_example['candidate_NPs']
            tmp_candidate_NPs += cdd_nps
            tmp_correct_candidate_NPs = pronoun_example['correct_NPs']

            if pronoun_example['not_discussed']:
                self.all_coref_not_discussed += len(tmp_correct_candidate_NPs)
            else:
                self.all_coref_discussed += len(tmp_correct_candidate_NPs)
           
            find_pronoun = False
            for coref_cluster in predicted_clusters:
                for mention in coref_cluster:
                    mention_start_index = mention[0]
                    if mention_start_index == tmp_pronoun_index:
                        find_pronoun = True
                if find_pronoun and pronoun_example['reference_type'] == 0:
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
                            if tmp_mention_span[0] < len(all_sentence) and\
                                tmp_mention_span[0] == tmp_mention_span[1] and\
                                all_sentence[tmp_mention_span[0]] in self.pronoun_list and\
                                len(tmp_candidate_NPs[matched_np_id]) > 1:
                                continue
                            matched_cdd_np_ids.append(matched_np_id)
                            self.predict_coreference += 1
                            if pronoun_example['not_discussed']:
                                self.predict_coref_not_discussed += 1
                            else:
                                self.predict_coref_discussed += 1
                            matched_np_id = util.verify_correct_NP_match(tmp_mention_span, tmp_correct_candidate_NPs, 'cover', matched_crr_np_ids)
                            if matched_np_id is not None:
                                matched_crr_np_ids.append(matched_np_id)
                                self.correct_predict_coreference += 1
                                if pronoun_example['not_discussed']:
                                    self.correct_predict_coref_not_discussed += 1
                                else:
                                    self.correct_predict_coref_discussed += 1
                    break

            self.all_coreference += len(tmp_correct_candidate_NPs)

