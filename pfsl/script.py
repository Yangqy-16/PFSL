from subprocess import call
import json
import os
import itertools
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

input_dir = '/data/qingyuyang/SL/MVGCNiSL/sl_data/specific'
output_dir = './output'

cell_lines = ['A549'] #, 'MELJUSO', 'HS944T'
scenes = ['C1'] #, 'C2', 'Tail'  'AKT1' Tail

for cl, scene in list(itertools.product(cell_lines, scenes)):
    # if not os.path.exists(f'../sl_data/reg/{cl}/{scene}'):  # specific case
    #     continue
    print(f'=================== {cl} / {scene} ===================')

    result_path = f'{output_dir}/{cl}/{scene}'
    os.makedirs(result_path, exist_ok=True)

    column_names = ['Accuracy', 'Precision', 'Recall', 'AUC', 'AUPR', 'F1', 'BACC']
    metric_df = pd.DataFrame(0, index=range(6), columns=column_names)

    for fold in range(5):
        print(f'=================== Fold {fold} ===================')

        call([
            'python', 'train.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--cell_line', cl,
            '--scene', scene,
            '--fold', str(fold)
        ])
        
        with open(f'{output_dir}/{cl}/{scene}/{fold}.json', 'r') as f:
            results = json.load(f)

        test_metrics = [results['acc'], results['precision'], 
                        results['recall'], results['auroc'], 
                        results['auprc'], results['f1'], results['bacc']]
        metric_df.iloc[fold] = test_metrics

    mean_values = metric_df.iloc[0:5].mean()
    std_values = metric_df.iloc[0:5].std()
    new_row = [f"{mean:.4f} ({std:.4f})" for mean, std in zip(mean_values, std_values)]
    metric_df.iloc[5] = new_row
    metric_df.to_csv(f'{result_path}/cv_results.csv', index=False)
