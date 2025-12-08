import os
import pandas as pd
from argparse import ArgumentParser
import yaml
import warnings
from tqdm import tqdm
from datetime import datetime, timezone
warnings.warn = lambda *args,**kwargs: None

def process_class_stats(file_path):
    columns = ['class_name', 'num_files', 'num_objects', 'precision', 'recall', 'map50', 'map']

    # Read the text file into a pandas DataFrame
    # df = pd.read_csv(file_path, delim_whitespace=True)
    warnings.simplefilter# / warnings.catch_warnings
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, header=None)

    # Find the index where the last repetition of 'all' starts
    last_all_index = df[df['class_name'] == 'all'].index[-1]
    max_map = max(df[df['class_name'] == 'all']['map50'])
    # Slice the DataFrame from the last 'all' row downward
    sliced_df = df.iloc[last_all_index:]

    # Create a new DataFrame with num_objects and renamed map50 columns
    result_df = sliced_df[['class_name', 'num_objects', 'map50']].copy()
    result_df = result_df.set_index('class_name')
    # Rename the 'map50' column to 'class_name_map50' for each class_name
    # result_df['class_name_map50'] = result_df['class_name'] + '_map50'
    result_df = result_df[['map50']].T

    # Rename the columns by appending '_map50'
    result_df.columns = [f"{col}_map50" for col in result_df.columns]
    # Select only the required columns
    # result_df = result_df[['num_objects', 'class_name_map50']]

    # Write the result to a CSV file
    # result_df.to_csv(output_csv, index=False)

    return result_df, max_map

def main(args: list = None):
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/hanoch/projects/tir_od/runs/train', metavar='PATH',
                        help="if given, all output of the training will be in this folder. "
                             "The exception is the tensorboard logs.")

    parser.add_argument('--task', default='train', help='train, val, test, speed or study')

    args = parser.parse_args(args)
    if 0:
        path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/test_eileen_best_qual/csv'
        filenames = [os.path.join(path, x) for x in os.listdir(path)
                     if x.endswith('csv')]

        df_acm = pd.DataFrame()
        for file in filenames:
            df = pd.read_csv(file, index_col=False)
            file_patt = df.full_file_name[0].split('/')[-1].split('.')[0].split('_')[1:]
            df['file_name'] = file_patt[0] + '_' +  "_".join(df.full_file_name[0].split('/')[-1].split('.')[0].split('_')[1:])
            df['val'] = 0
            df_acm = df_acm.append((df))

        cols = df_acm.columns.to_list()
        cols2 = [cols[-2]] + cols[2:-2] + [cols[-1]]
        cols3 = cols2[:-3] + cols2[-2:]
        df_acm = df_acm[cols3]
        df_acm.to_csv(os.path.join(path, 'merged.csv'), index=False)

    else:
        path = args.path
        path_result = '/home/hanoch/projects/tir_od'
        results_columns = ['Epoch', 'gpu_mem', 'box_loss', 'obj_loss', 'cls_loss', 'total_loss', 'labels', 'img_size',
                           'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'val_box_loss', 'val_obj_loss', 'val_cls_loss']

        # from pathlib import Path
        # Path(os.path.join(path, 'merged')).mkdir(parents=True, exist_ok=True)
        #
        # filenames = [os.path.join(path, x) for x in os.listdir(path)
        #              if x.endswith('csv')]
        #
        # df_acm = pd.DataFrame()
        # for file in filenames:
        #     df = pd.read_csv(file, index_col=False)
        #     if 1:
        #         df.columns = df.iloc[0]
        #         df = df[1:2]
        #     print(file)
        #     df_acm = df_acm.append((df))
        #
        # # df_acm = df_acm.reindex(sorted(df_acm.columns), axis=1)
        # df_acm.to_csv(os.path.join(path, 'merged', 'merged.csv'), index=False)
        # List to hold the data
        data = []
        root_dir = path
        # Iterate through all the subfolders
        for subdir, dirs, files in tqdm(os.walk(root_dir)):
            # Check if 'results.txt' and 'hyp.yaml' exist in the current subdir
            daytime_str = datetime.fromtimestamp(os.stat(subdir).st_ctime).strftime('%Y-%m-%d %H:%M')
            results_path = os.path.join(subdir, 'results.txt')
            hyp_path = os.path.join(subdir, 'hyp.yaml')
            opt_path = os.path.join(subdir, 'opt.yaml')
            per_class_results = os.path.join(subdir, 'class_stats.txt')
            best_fitness_path = os.path.join(subdir, 'best_fitness.txt')

            if os.path.exists(results_path) and os.path.exists(hyp_path) and os.path.exists(opt_path):
                # Get the last line from 'results.txt'
                with open(results_path, 'r') as results_file:
                    last_line = results_file.readlines()[-1].strip()

                # Split the last line into the corresponding fields
                results_values = last_line.split()

                # Ensure that the last line contains the expected number of fields
                if len(results_values) == len(results_columns):
                    results_data = dict(zip(results_columns, results_values))
                else:
                    print(f"Warning: Unexpected format in {results_path}, skipping.")
                    continue

                # Load the 'hyp.yaml' file
                with open(hyp_path, 'r') as hyp_file:
                    hyp_data = yaml.safe_load(hyp_file)

                with open(opt_path, 'r') as opt_file:
                    opt_data = yaml.safe_load(opt_file)

                df_per_class_results = pd.DataFrame()
                if os.path.exists(per_class_results):
                    df_per_class_results, max_map = process_class_stats(per_class_results)

                # Add the result and the 'hyp.yaml' content into the data list
                row = {
                    'subdir': subdir,
                }
                # Update the row with the parsed results.txt values
                row.update(results_data)
                # Update the row with the hyperparameters from the 'hyp.yaml'
                row.update(hyp_data)

                row.update(opt_data)
                if not df_per_class_results.empty:
                    row.update(df_per_class_results.to_dict(orient='list'))
                    row.update({'max_map50': max_map})

                row.update({'daytime_str': daytime_str})
                data.append(row)


            if os.path.exists(best_fitness_path):
                with open(best_fitness_path, 'r') as results_file:
                    best_fitness = results_file.readlines()[-1].strip()
                row.update({'best_map50': best_fitness})
                data.append(row)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['daytime_str'])
        df = df.sort_values(by='date')

        # Save the DataFrame to a CSV file
        output_csv = 'runs_' + str(args.task) + '_summary.csv'
        # df.to_csv(output_csv, index=False)
        df.to_csv(os.path.join(path_result, output_csv), index=False)

        print(f"Data successfully written to {os.path.join(path_result, output_csv)}")

if __name__ == '__main__':
    main()

