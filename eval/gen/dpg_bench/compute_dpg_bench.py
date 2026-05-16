# Copyright 2024 Tencent QQGY Lab
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from:
# https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench

import argparse
import json
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument("--image-root-path", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--csv", type=str, required=True, help="Path to the official DPG-Bench question CSV.")
    parser.add_argument("--res-path", type=str, default=None)
    parser.add_argument("--pic-num", type=int, default=4)
    parser.add_argument("--vqa-model", type=str, default="mplug")
    parser.add_argument(
        "--vqa-model-path",
        type=str,
        default=os.environ.get(
            "DPG_SCORE_VQA_MODEL_PATH",
            "damo/mplug_visual-question-answering_coco_large_en",
        ),
        help="Local path or ModelScope model id for the mPLUG VQA model. Can also be set by DPG_SCORE_VQA_MODEL_PATH.",
    )
    return parser.parse_args()


def get_process_info():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def make_rank_paths(res_path, rank):
    stem, ext = osp.splitext(res_path)
    ext = ext or ".txt"
    return {
        "result": f"{stem}.rank{rank}{ext}",
        "detail": f"{stem}.rank{rank}_detail.txt",
        "summary": f"{stem}.rank{rank}.json",
    }


def detail_path_for(res_path):
    stem, ext = osp.splitext(res_path)
    if ext == ".txt":
        return f"{stem}_detail.txt"
    return f"{res_path}_detail.txt"


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt, device="gpu"):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {"image": image, "question": question}
        result = self.pipeline_vqa(input_vqa)
        return result["text"]


def prepare_dpg_data(csv_path):
    previous_id = ""
    current_id = ""
    question_dict = {}

    # Columns:
    # item_id, text, keywords, proposition_id, dependency, category_broad,
    # category_detailed, tuple, question_natural_language
    data = pd.read_csv(csv_path)
    for row_idx, line in data.iterrows():
        # Keep official behavior from ELLA.
        if row_idx == 0:
            continue

        current_id = str(line.item_id)
        qid = int(line.proposition_id)
        dependency_list_int = [int(d.strip()) for d in str(line.dependency).split(",")]

        if current_id == previous_id:
            question_dict[current_id]["qid2tuple"][qid] = line.tuple
            question_dict[current_id]["qid2dependency"][qid] = dependency_list_int
            question_dict[current_id]["qid2question"][qid] = line.question_natural_language
        else:
            question_dict[current_id] = {
                "qid2tuple": {qid: line.tuple},
                "qid2dependency": {qid: dependency_list_int},
                "qid2question": {qid: line.question_natural_language},
            }
        previous_id = current_id

    return question_dict


def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image
    return input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))


def build_crop_tuples(resolution, pic_num):
    crop_tuples_list = [
        (0, 0, resolution, resolution),
        (resolution, 0, resolution * 2, resolution),
        (0, resolution, resolution, resolution * 2),
        (resolution, resolution, resolution * 2, resolution * 2),
    ]
    if pic_num <= 1:
        return [None]
    return crop_tuples_list[:pic_num]


def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path).convert("RGB")
    crop_tuples = build_crop_tuples(resolution, args.pic_num)

    key = osp.splitext(osp.basename(image_path))[0]
    value = question_dict.get(key, None)
    if value is None:
        raise KeyError(f"No DPG question entry found for image key '{key}' in {args.csv}")

    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]
    qid2scores_orig = {}
    scores = []

    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        qid2scores = {}

        for qid, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2scores[qid] = float(answer == "yes")
            with open(args.rank_detail_path, "a", encoding="utf-8") as f:
                f.write(
                    image_path
                    + ", "
                    + str(crop_tuple)
                    + ", "
                    + question
                    + ", "
                    + answer
                    + "\n"
                )

        qid2scores_orig = qid2scores.copy()
        for qid, parent_ids in qid2dependency.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[qid] = 0

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)

    average_score = sum(scores) / len(scores)
    with open(args.rank_res_path, "a", encoding="utf-8") as f:
        f.write(image_path + ", " + ", ".join(str(i) for i in scores) + ", " + str(average_score) + "\n")

    return average_score, qid2tuple, qid2scores_orig


def iter_image_files(image_root_path):
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    filenames = []
    for filename in os.listdir(image_root_path):
        image_path = osp.join(image_root_path, filename)
        if not osp.isfile(image_path):
            continue
        if osp.splitext(filename)[1].lower() not in valid_exts:
            continue
        filenames.append(filename)
    return sorted(filenames)


def write_rank_summary(path, scores, category2scores):
    payload = {
        "scores": scores,
        "category2scores": {category: values for category, values in category2scores.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def read_rank_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["scores"], payload["category2scores"]


def append_file(src_path, dst_file):
    if not osp.exists(src_path):
        return
    with open(src_path, "r", encoding="utf-8") as src:
        for line in src:
            dst_file.write(line)


def main():
    args = parse_args()
    rank, local_rank, world_size = get_process_info()

    if not osp.isdir(args.image_root_path):
        raise FileNotFoundError(f"Image root path not found: {args.image_root_path}")
    if not osp.isfile(args.csv):
        raise FileNotFoundError(f"DPG-Bench CSV not found: {args.csv}")

    question_dict = prepare_dpg_data(args.csv)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    if args.res_path is None:
        args.res_path = osp.join(args.image_root_path, f"dpg-bench_{timestamp}_results.txt")

    res_dir = osp.dirname(args.res_path)
    if res_dir:
        os.makedirs(res_dir, exist_ok=True)

    rank_paths = make_rank_paths(args.res_path, rank)
    args.rank_res_path = rank_paths["result"]
    args.rank_detail_path = rank_paths["detail"]
    with open(args.rank_res_path, "w", encoding="utf-8") as f:
        pass
    with open(args.rank_detail_path, "w", encoding="utf-8") as f:
        pass

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    if args.vqa_model == "mplug":
        vqa_model = MPLUG(ckpt=args.vqa_model_path, device=device)
    else:
        raise NotImplementedError(f"Unsupported VQA model: {args.vqa_model}")

    filename_list = iter_image_files(args.image_root_path)
    local_filename_list = filename_list[rank::world_size]

    local_scores = []
    local_category2scores = defaultdict(list)
    model_id = osp.basename(osp.normpath(args.image_root_path))
    if rank == 0:
        print(f"Start to conduct evaluation of {model_id}")
        print(f"Image root: {args.image_root_path}")
        print(f"World size: {world_size}")

    print(f"Rank {rank}: processing {len(local_filename_list)} / {len(filename_list)} images on {device}", flush=True)
    for filename in tqdm(local_filename_list, disable=(rank != 0)):
        image_path = osp.join(args.image_root_path, filename)
        try:
            score, qid2tuple, qid2scores = compute_dpg_one_sample(
                args=args,
                question_dict=question_dict,
                image_path=image_path,
                vqa_model=vqa_model,
                resolution=args.resolution,
            )
            local_scores.append(score)
            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split("(")[0].strip()
                local_category2scores[category].append(qid2scores[qid])
        except Exception as exc:
            print("Failed filename:", filename, exc)
            continue

    write_rank_summary(rank_paths["summary"], local_scores, local_category2scores)

    if rank == 0:
        all_rank_paths = [make_rank_paths(args.res_path, rank_idx) for rank_idx in range(world_size)]
        while not all(osp.exists(paths["summary"]) for paths in all_rank_paths):
            time.sleep(5)

        global_dpg_scores = []
        global_category2scores = defaultdict(list)
        for paths in all_rank_paths:
            rank_scores, rank_category2scores = read_rank_summary(paths["summary"])
            global_dpg_scores.extend(rank_scores)
            for category, values in rank_category2scores.items():
                global_category2scores[category].extend(values)

        mean_dpg_score = np.mean(global_dpg_scores) if global_dpg_scores else 0.0
        global_categories = set(global_category2scores.keys())

        global_category2scores_l1 = defaultdict(list)
        for category in global_categories:
            l1_category = category.split("-")[0].strip()
            global_category2scores_l1[l1_category].extend(global_category2scores[category])

        final_detail_path = detail_path_for(args.res_path)
        with open(args.res_path, "w", encoding="utf-8") as result_file:
            for paths in all_rank_paths:
                append_file(paths["result"], result_file)
        with open(final_detail_path, "w", encoding="utf-8") as detail_file:
            for paths in all_rank_paths:
                append_file(paths["detail"], detail_file)

        output = f"Model: {model_id}\n"
        output += "L1 category scores:\n"
        for l1_category in sorted(global_category2scores_l1.keys()):
            output += f"\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n"
        output += "L2 category scores:\n"
        for category in sorted(global_categories):
            output += f"\t{category}: {np.mean(global_category2scores[category]) * 100}\n"
        output += f"Image path: {args.image_root_path}\n"
        output += f"Question CSV: {args.csv}\n"
        output += f"Save results to: {args.res_path}\n"
        output += f"DPG-Bench score: {mean_dpg_score * 100}"
        with open(args.res_path, "a", encoding="utf-8") as f:
            f.write(output + "\n")
        print(output)

        for paths in all_rank_paths:
            for path in paths.values():
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
