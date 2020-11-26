# @File : epic_data_analysis.py 
# @Time : 2019/10/4 
# @Email : jingjingjiang2017@gmail.com 

import matplotlib.pyplot as plt
import seaborn as sns

from data.epic import *

# bounding box: (<top:int>,<left:int>,<height:int>,<width:int>).
anno_path_ = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/data/annotations'
annos = pd.read_csv(os.path.join(anno_path_, 'EPIC_train_object_labels.csv'),
                    converters={"bounding_boxes": literal_eval})


def plot_classes(data_df, feature, fs=8, show_percents=True,
                 color_palette='Set3'):
    f, ax = plt.subplots(1, 1, figsize=(2 * fs, 4))
    total = float(len(data_df))
    g = sns.countplot(data_df[feature],
                      order=data_df[feature].value_counts().index,
                      palette=color_palette)
    g.set_title(f"Number and percentage of labels for each class of {feature}")
    if show_percents:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    '{:1.2f}%'.format(100 * height / total),
                    ha="center")
    plt.show()


if __name__ == '__main__':
    train_dataset = EpicSequenceDataset(args)
    # train_dataloader = DataLoader(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  num_workers=3, shuffle=True)
    sequence_lens = []
    for i, data in enumerate(train_dataloader):
        img, mask = data
        sequence_lens.append(img.shape[0])

