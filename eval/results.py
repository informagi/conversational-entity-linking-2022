from itertools import combinations
import re
import json
import pandas as pd


class Metrics():
    def __init__(self, runs, ments_already_normalized=True):
        """
        Args:
            runs: a list of tuples (name_of_the_run, dataset_type, path_to_the_run_file)
            ments_already_normalized: if True, mentions are already normalized. If False, mentions are normalized here
        """
        # A dictionary with dataset_type as keys and file_to_gold_data as values
        gold_dc = {
            'Test': './gold/Test/gold.json',
            'Val': './gold/Val/gold.json',
            'ConEL21-PE': './gold/ConEL21-PE/gold.json',
        }
        self.sanity_test(runs, gold_dc)
        self.runs = runs
        self.gold_dc = {k:json.load(open(v)) for k,v in gold_dc.items()}
        self.ments_already_normalized = ments_already_normalized # If you want to evaluate your own predictions, this should be set as False

    def sanity_test(self,runs, gold_dc):
        for run in runs:
            dataset_type = run[1]
            assert dataset_type in gold_dc.keys(), 'Dataset type {} not found in gold data'.format(dataset_type)
        print('Passed sanity test')

    def norm_ment(self, ment):
        """Normalize mention"""
        ment = re.sub('^a ', '', ment)
        ment = re.sub('^A ', '', ment)
        ment = re.sub('^an ', '', ment)
        ment = re.sub('^An ', '', ment)
        ment = re.sub('^the ', '', ment)
        ment = re.sub('^The ', '', ment)
        return ment

    def calc_metrics(self, pred, gold):
        """Calculate precision, recall, and F1 metrics"""
        
        pred = set(tuple(x) for x in pred)
        gold = set(tuple(x) for x in gold)

        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)

        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = tp/(tp+(fp+fn)/2)

        return prec, recall, f1

    def create_table(self):
        """Create results table from run files"""

        table = {}
        
        for run in self.runs:
            run_name, dataset_type, run_file = run # E.g., ('REL', 'ConEL21-PE', './runs/ConEL21-PE/REL-WO-PE.json')
            print(f'Reading {run_file}')
            pred = json.load(open(run_file))
            gold = self.gold_dc[dataset_type]

            # If mentions are not normalized, then they are normalized here
            # NOTE: The mentions in the provided run files have already normalized
            if not self.ments_already_normalized:
                pred = [[document_id, turn_num, self.norm_ment(ment), ent] for document_id, turn_num, ment, ent in pred]
                gold = [[document_id, turn_num, self.norm_ment(ment), ent] for document_id, turn_num, ment, ent in gold]

            prf_dc = {}
            prf_dc['P'], prf_dc['R'], prf_dc['F'] = self.calc_metrics(pred, gold)

            for prf in ['P', 'R', 'F']:
                if (dataset_type, prf) not in table.keys():
                    table[(dataset_type, prf)] = {}
                table[(dataset_type, prf)][run_name] = prf_dc[prf]

        table = pd.DataFrame(table)
        return table