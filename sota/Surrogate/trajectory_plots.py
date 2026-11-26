import json
import numpy as np
import pandas as pd

import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns


def surrogate_data_plots(res_dir, sel = False):
    #load res_dir/candiate_log_trail_0_seed_*.json where * can be anything you  want t
    files = glob.glob(os.path.join(res_dir, 'candidate_log_trial_0_seed_*.json'))
    print(res_dir, files)
    data = []
    for file in files:
        with open(file, 'r') as f:
            #retrieve generation, predicted accuracy, true accuracy for each individual (and this json file is a list of individuals)
            trial_data = json.load(f)
            for entry in trial_data:
                data.append({
                    'epoch': entry['generation'],
                    'true_accuracy': entry['true_accuracy'],
                    'predicted_accuracy': entry['predicted_accuracy'],
                    'selected': entry['selected']
                })
        #filter by selected

        data = [d for d in data if str(d.get('selected')).lower() == "true" or not sel]
        #create a plot of predicted vs true accuracy across generations (x-axis is generation, y-axis is accuracy error) and save it as res_dir/surrogate_accuracy_by_gen.png
        df = pd.DataFrame(data)
        for col in ('epoch', 'true_accuracy', 'predicted_accuracy'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['epoch', 'true_accuracy', 'predicted_accuracy'])
        if df.empty:
            print(f"[trajectory_plots] No numeric rows to plot in {res_dir} (sel={sel})")
            return
        df['accuracy_error'] = np.abs(df['predicted_accuracy'] - df['true_accuracy'])/df['true_accuracy']
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='accuracy_error', data=df)
        plt.xlabel('Generation')
        plt.ylabel('Relative Accuracy Error')
        plt.title('Surrogate Accuracy Error Across Generations (Absolute Value of (Predicted - True) / True)')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'].to_numpy(dtype=float),
                                      df['accuracy_error'].to_numpy(dtype=float), 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()

        plt.savefig(os.path.join(res_dir, f'surrogate_accuracy_by_gen_{str(sel)}.png'))
        plt.close()
        #create 2 plots of preductaed and true accuracy across generations (x-axis is generation, y-axis is accuracy) and save it as res_dir/surrogate_pred_true_by_gen.png
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='predicted_accuracy', data=df, label='True Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Average Predicted Accuracy')
        plt.title('Surrogate Predicted Accuracy Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'].to_numpy(dtype=float),
                                      df['predicted_accuracy'].to_numpy(dtype=float), 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'surrogate_predict_accuracy_by_gen_{str(sel)}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='true_accuracy', data=df, label='True Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Average True Accuracy')
        plt.title('Surrogate True Accuracy Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'].to_numpy(dtype=float),
                                      df['true_accuracy'].to_numpy(dtype=float), 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'surrogate_true_accuracy_by_gen_{str(sel)}.png'))
        plt.close()

        #plot variance of accuracy error across generations (x-axis is generation, y-axis is variance of accuracy error) and save it as res_dir/surrogate_accuracy_error_variance_by_gen.png
        df['accuracy_error'] = np.abs(df['predicted_accuracy'] - df['true_accuracy'])/df['true_accuracy']
        accuracy_error_variance = df.groupby('epoch')['accuracy_error'].var().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='accuracy_error', data=accuracy_error_variance)
        plt.xlabel('Generation')
        plt.ylabel('Variance of Relative Accuracy Error')
        plt.title('Variance of Surrogate Accuracy Error Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(accuracy_error_variance['epoch'], accuracy_error_variance['accuracy_error'], 1)
        plt.plot(accuracy_error_variance['epoch'], slope * accuracy_error_variance['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'surrogate_accuracy_error_variance_by_gen_{str(sel)}.png'))
        plt.close()