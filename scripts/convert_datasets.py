import argparse
import gzip
import io
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--source_split', type=str, default='val')
parser.add_argument('--target_split', type=str, default='val_mt')
parser.add_argument('--dataset_path', type=str,
                    default='data/datasets/pointnav/gibson/v1/')
parser.add_argument('--num_episodes_per_scene', type=int, default=0,
                    help="0 for all")
parser.add_argument('--split_by_size', type=int, default=0)
parser.add_argument('--multi_thread', type=int, default=1)
parser.add_argument('--scene_name', type=str, default="0")

args = parser.parse_args()

scenes = {}
split_names = ["small", "large"]
scenes[split_names[1]] = ['Cantwell', 'Eastville', 'Mosquito', 'Scioto']
scenes[split_names[0]] = ['Denmark', 'Edgemere', 'Elmira', 'Eudora',
                          'Greigsville', 'Pablo', 'Ribera', 'Sands',
                          'Sisters', 'Swormville']

if args.scene_name != "0":
    split_names = [args.scene_name]
    scenes[split_names[0]] = [args.scene_name]

source_dataset_path = "{path}/{split}/{split}.json.gz".format(
    path=args.dataset_path, split=args.source_split)

assert os.path.exists(source_dataset_path), "Invalid dataset path"

with gzip.open(source_dataset_path, "rt") as f:
    deserialized = json.loads(f.read())

data = {}
for episode in deserialized['episodes']:
    scene = episode['scene_id'].split("/")[-1].split(".")[0]
    if scene in data.keys():
        if len(data[scene]['episodes']) < args.num_episodes_per_scene or \
                args.num_episodes_per_scene == 0:
            data[scene]['episodes'].append(episode)
    else:
        data[scene] = {}
        data[scene]['episodes'] = [episode]


print("Dataset processesed")
print("Number of scenes: {}".format(len(data.keys())))
for scene in data.keys():
    print("{}: {} episodes".format(scene, len(data[scene]['episodes'])))


if args.multi_thread:
    if not args.split_by_size:
        target_dataset_folder = "{path}/{split}/".format(
            path=args.dataset_path, split=args.target_split)
        print("Writing dataset to {}".format(target_dataset_folder))

        target_dataset_content_folder = "{}/content/".format(
            target_dataset_folder)

        if not os.path.exists(target_dataset_content_folder):
            os.makedirs(target_dataset_content_folder)

        for scene in data.keys():
            outfilename = target_dataset_content_folder + scene + ".json.gz"
            with gzip.open(outfilename, 'wb') as output:
                with io.TextIOWrapper(output, encoding='utf-8') as enc:
                    enc.write(json.dumps(data[scene]))

        outfilename = target_dataset_folder \
                        + "{}.json.gz".format(args.target_split)
        with gzip.open(outfilename, 'wb') as output:
            with io.TextIOWrapper(output, encoding='utf-8') as enc:
                enc.write("{\"episodes\": []}")

    else:
        for i in range(len(split_names)):
            split = args.target_split + "_" + split_names[i]
            target_dataset_folder = "{path}/{split}/".format(
                path=args.dataset_path, split=split)
            print("Writing dataset to {}".format(target_dataset_folder))

            target_dataset_content_folder = "{}/content/".format(
                target_dataset_folder)

            if not os.path.exists(target_dataset_content_folder):
                os.makedirs(target_dataset_content_folder)

            for scene in data.keys():
                if scene in scenes[split_names[i]]:
                    outfilename = target_dataset_content_folder \
                                    + scene + ".json.gz"
                    with gzip.open(outfilename, 'wb') as output:
                        with io.TextIOWrapper(output, encoding='utf-8') as enc:
                            enc.write(json.dumps(data[scene]))

            outfilename = target_dataset_folder + "{}.json.gz".format(split)
            with gzip.open(outfilename, 'wb') as output:
                with io.TextIOWrapper(output, encoding='utf-8') as enc:
                    enc.write("{\"episodes\": []}")


else:
    if not args.split_by_size:
        target_dataset_folder = "{path}/{split}/".format(
            path=args.dataset_path, split=args.target_split)

        print("Writing dataset to {}".format(target_dataset_folder))

        if not os.path.exists(target_dataset_folder):
            os.makedirs(target_dataset_folder)

        data_combined = {}
        data_combined['episodes'] = []
        for scene in data.keys():
            for episode in data[scene]['episodes']:
                data_combined['episodes'].append(episode)

        outfilename = target_dataset_folder \
                        + "{}.json.gz".format(args.target_split)
        with gzip.open(outfilename, 'wb') as output:
            with io.TextIOWrapper(output, encoding='utf-8') as enc:
                enc.write(json.dumps(data_combined))

    else:
        for i in range(len(split_names)):
            split = args.target_split + "_" + split_names[i]
            target_dataset_folder = "{path}/{split}/".format(
                path=args.dataset_path, split=split)

            print("Writing dataset to {}".format(target_dataset_folder))

            if not os.path.exists(target_dataset_folder):
                os.makedirs(target_dataset_folder)

            data_combined = {}
            data_combined['episodes'] = []
            for scene in data.keys():
                if scene in scenes[split_names[i]]:
                    for episode in data[scene]['episodes']:
                        data_combined['episodes'].append(episode)

            outfilename = target_dataset_folder + "{}.json.gz".format(split)
            with gzip.open(outfilename, 'wb') as output:
                with io.TextIOWrapper(output, encoding='utf-8') as enc:
                    enc.write(json.dumps(data_combined))
