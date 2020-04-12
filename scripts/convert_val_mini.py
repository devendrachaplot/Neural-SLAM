import argparse
import os
import json
import gzip
import io

parser = argparse.ArgumentParser()
parser.add_argument('--source_split', type=str, default='val_mini')
parser.add_argument('--target_split', type=str, default='val_mini_c')
parser.add_argument('--dataset_path', type=str, default='../data/datasets/pointnav/gibson/v1/')
parser.add_argument('--num_episodes_per_scene', type=int, default=0,
                    help="0 for all")
parser.add_argument('--split_by_size', type=int, default=1)
parser.add_argument('--multi_thread', type=int, default=0)
parser.add_argument('--scene_name', type=str, default="0")


args = parser.parse_args()

scenes = {}

source_dataset_path = "{path}/{split}/{split}.json.gz".format(
                            path=args.dataset_path, split=args.source_split)

assert os.path.exists(source_dataset_path), "Invalid dataset path: {}".format(source_dataset_path)

with gzip.open(source_dataset_path, "rt") as f:
    deserialized = json.loads(f.read())

data = {}
for episode in deserialized['episodes']:
    scene = episode['scene_id'].split("/")[-1].split(".")[0]
    episode['scene_id'] = episode['scene_id'].replace("/habitat-challenge-data","data/scene_datasets")
    if scene in data.keys():
        if len(data[scene]['episodes']) < args.num_episodes_per_scene or \
                args.num_episodes_per_scene == 0:
            data[scene]['episodes'].append(episode)
    else:
        data[scene] = {}
        data[scene]['episodes'] = [episode]

target_dataset_folder = "{path}/{split}/".format(
                path=args.dataset_path, split=args.target_split)

if not os.path.exists(target_dataset_folder):
    os.makedirs(target_dataset_folder)


data_combined = {}
data_combined['episodes'] = []
for scene in data.keys():
    for episode in data[scene]['episodes']:
        data_combined['episodes'].append(episode)

outfilename = target_dataset_folder+"{}.json.gz".format(args.target_split)
with gzip.open(outfilename, 'wb') as output:
    with io.TextIOWrapper(output, encoding='utf-8') as enc:
        enc.write(json.dumps(data_combined))
