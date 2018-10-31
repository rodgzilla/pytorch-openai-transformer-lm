import os
import pathlib
import pandas as pd
import tqdm

def extract_paths(
        data_folder,
        dataset,
        sent
):
    """
    dataset = 'train' or 'test'
    sent = 'pos' or 'neg'
    """
    folder_path = pathlib.Path(data_folder) / dataset / sent
    return [folder_path / fn
            for fn in os.listdir(folder_path)]

def get_comment_list(paths, desc):
    comments = []
    for comment_file_path in tqdm.tqdm(paths, desc = desc, ncols = 80):
        with open(comment_file_path, 'r') as comment_file:
            comment = comment_file.read()
            comment = comment.replace('<br />', ' ')
            comment = comment.replace('\\', '')
            comments.append(comment)

    return comments

def create_df_dataset(dataset):
    data_folder = 'data/aclImdb/'
    paths       = extract_paths(data_folder, dataset, 'neg')
    comments    = get_comment_list(paths, 'Training set')
    neg_records = [(comment, 0) for comment in comments]
    paths       = extract_paths(data_folder, dataset, 'pos')
    comments    = get_comment_list(paths, 'Test set')
    pos_records = [(comment, 1) for comment in comments]
    df          = pd.DataFrame.from_records(
        neg_records + pos_records,
        columns = ['comment', 'sentiment']
    )

    return df


def main():
    df_train = create_df_dataset('train')
    df_train.to_csv('data/aclImdb/train.csv', index = False)
    df_test = create_df_dataset('test')
    df_test.to_csv('data/aclImdb/test.csv', index = False)

if __name__ == '__main__':
    main()
