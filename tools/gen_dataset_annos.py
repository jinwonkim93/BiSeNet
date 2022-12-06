
import os
import os.path as osp
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def gen_coco():
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    root_path = './datasets/coco'
    save_path = './datasets/coco/'
    for mode in ('train', 'val'):
        im_root = osp.join(root_path, f'images/{mode}2017')
        lb_root = osp.join(root_path, f'labels/{mode}2017')

        ims = os.listdir(im_root)
        lbs = os.listdir(lb_root)

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{mode}2017/{name}.jpg,labels/{mode}2017/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))


def gen_ade20k():
    '''
        root_path:
            |- images
                |- training
                |- validation
            |- annotations
                |- training
                |- validation
    '''
    root_path = './datasets/ade20k/'
    save_path = './datasets/ade20k/'
    folder_map = {'train': 'training', 'val': 'validation'}
    for mode in ('train', 'val'):
        folder = folder_map[mode]
        im_root = osp.join(root_path, f'images/{folder}')
        lb_root = osp.join(root_path, f'annotations/{folder}')

        ims = os.listdir(im_root)
        lbs = os.listdir(lb_root)

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{folder}/{name}.jpg,annotations/{folder}/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))


def gen_facesynthetic():
    '''
        root_path:
            |- dataset-100000
                |- 000000.png
                |- 000000_seg.png
                
    '''
    root_path = './datasets/facesynthetic/'
    save_path = './datasets/facesynthetic/'
    folder_map = {'train': 'dataset_100000'}
    mode = 'train'
    folder = folder_map[mode]
    im_root = osp.join(root_path, f'{folder}')
    files = list(Path(im_root).glob("*.png"))

    print(len(files)) #image, seg image total 20k
    ims = []
    lbs = []
    for file in files:
        if "seg" in file.name:
            lbs.append(file)
        else:
            ims.append(file)

    im_names = [el.stem for el in ims]
    lb_names = [el.stem.replace('_seg', '') for el in lbs]
    common_names = list(set(im_names) & set(lb_names))
    lines = [
        f'{folder}/{name}.png,{folder}/{name}_seg.png'
        for name in common_names
    ]
    lines_train, lines_val = train_test_split(lines, test_size=0.03, random_state=2022)
    with open(f'{save_path}/{mode}.txt', 'w') as fw:
        fw.write('\n'.join(lines_train))
    with open(f'{save_path}/val.txt', 'w') as fw:
        fw.write('\n'.join(lines_val))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', dest='dataset', type=str, default='coco')
    args = parse.parse_args()

    if args.dataset == 'coco':
        gen_coco()
    elif args.dataset == 'ade20k':
        gen_ade20k()
    elif args.dataset == 'facesynthetic':
        gen_facesynthetic()
