from subprocess import call
import os
import numpy as np
import pandas as pd
import shutil
import itertools
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1' #0,1,2

output_root = '/data/qingyuyang/SL/PFSL/output/pfsl_updated'

cell_lines = ['MELJUSO', 'A549', 'HS944T'] #'HSC5', 'A549' || NOTE: 'NSMexp_indeptest_1:1', 'NSMexp_indeptest_1:5' HS944T A549 IPC298 MEL202 MEWO PK1 HELA A375 GI1 RPE1 PATU8988S HS936T HS944T HSC5 NSMexp_indeptest_1:1
scenes = ['C1', 'C2', 'Tail'] #

# use_aspects = list(range(0, 24)) #['null'] + 

for cell_line, scene in list(itertools.product(cell_lines, scenes)): #, use_func, use_aspects, unuse_func  
    print(f'======================== {cell_line} / {scene} ========================')
    # print(f'======================== use_func: {use_func} ========================')

    data_root = f'/data/qingyuyang/SL/PFSL/data/SLKB/specific_1:1/{cell_line}/{scene}'  # NOTE: specific synlethdb slbench SLKB/specific SLKB/case

    table_path = f'{output_root}/{cell_line}/{scene}' #/{use_func}  if unuse_func != 'null' else f'{output_root}/{cell_line}/{scene}'
    os.makedirs(table_path, exist_ok=True)

    column_names = ['Accuracy', 'Precision', 'Recall', 'AUC', 'AUPR', 'F1', 'BACC']
    metric_df = pd.DataFrame(0, index=range(6), columns=column_names)

    for test_fold in range(5):
        print(f'======================== Fold {test_fold} ========================')

        seed = 42
        call([
            'python', 'train.py', 
            '--config-file', 'pfsl/configuration/prosl.yaml', # NOTE:lora/config pf pfnew unified_token
            '--num-gpus', '1',
            'SEED', str(seed),
            'OUTPUT_DIR', f'{table_path}/fold_{test_fold}',
            'DATASET.MAX_LEN', '1000',
            'DATASET.TEST_FOLD', str(test_fold),
            'DATASET.TRAIN_FILE', f'{data_root}/sl_train_{test_fold}.csv',
            'DATASET.VAL_FILE', f'{data_root}/sl_val_{test_fold}.csv',
            'DATASET.TEST_FILE', f'{data_root}/sl_test_{test_fold}.csv',
            'DATASET.CL', cell_line,
            # 'MODEL.SEQ_USE_ATT', '0',
            # 'MODEL.FUNC_USE_ASPECT', str(use_func),
            # 'MODEL.FUNC_USE_ATT', 'null' if use_func == 'null' else True,
        ])

        newest_version = len(os.listdir(f'{table_path}/fold_{test_fold}/csv_log')) - 1
        result_csv_path = f'{table_path}/fold_{test_fold}/csv_log/version_{newest_version}/metrics.csv'
        result_csv = pd.read_csv(result_csv_path)
        test_result = result_csv.loc[len(result_csv) - 1]
        test_metrics = [test_result['test/acc'], test_result['test/precision'], 
                        test_result['test/recall'], test_result['test/auroc'], 
                        test_result['test/auprc'], test_result['test/f1'], test_result['test/bacc']]
        metric_df.iloc[test_fold] = test_metrics

        # shutil.copy(f'{table_path}/fold_{test_fold}/csv_log/version_{newest_version}/test_logits.csv',
        #             f'{table_path}/fold_{test_fold}_logits.csv')
        # shutil.copy(f'{table_path}/fold_{test_fold}/csv_log/version_{newest_version}/hparams.yaml',
        #             f'{table_path}/fold_{test_fold}_hparams.yaml')
        # shutil.rmtree(f'{table_path}/fold_{test_fold}')

    mean_values = metric_df.iloc[0:5].mean()
    std_values = metric_df.iloc[0:5].std()
    new_row = [f"{mean:.4f} ({std:.4f})" for mean, std in zip(mean_values, std_values)]
    metric_df.iloc[5] = new_row
    metric_df.to_csv(f'{table_path}/results.csv', index=False)
