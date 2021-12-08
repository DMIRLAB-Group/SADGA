import os
import sys

sys.path.append("./")
from evaluation import evaluation


def main(data_dir, res_dir, pred_name, acc_res_name):
    eval_steps = []
    for root, dirs, files in os.walk(res_dir):
        for f in files:
            file_ext = f.split('.')
            if file_ext[-1] == 'infer':
                step = int(file_ext[0][file_ext[0].index('step') + 4:])
                eval_steps.append(step)
    eval_steps.sort()

    ori = sys.stdout

    max_step, max_acc, cur_res = 0, 0, 0
    print('STEP', 'ACCURACY')
    for step in eval_steps:
        res_file = f"{res_dir}/{acc_res_name}_step{step}.txt"
        predict_file = f"{res_dir}/{pred_name}_step{step}.txt"
        if os.path.exists(res_file):
            with open(res_file) as f1:
                tmp = False
                for index, line1 in enumerate(f1.readlines()):
                    if tmp:
                        t = line1.split()
                        cur_res = t[-1]
                        break
                    if "EXACT MATCHING ACCURACY" in line1:
                        tmp = True
        else:
            scores = evaluation.main(gold=data_dir + 'dev_gold.sql', pred=predict_file,
                                     db_dir=data_dir + 'database/',
                                     table=data_dir + 'tables.json', etype='match', res_file=res_file)
            sys.stdout = ori
            cur_res = scores['all']['exact']

        if float(cur_res) >= float(max_acc):
            max_step, max_acc = step, cur_res
        print(step, cur_res)

    print("Best Result:", max_step, max_acc)