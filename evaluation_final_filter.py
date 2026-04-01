#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import os
import random
import time
import evaluate

from tqdm import tqdm
import datasets
import numpy as np
from typing import List, Dict
import re
import string
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter



tests = {
         'NER': [
                 "NCBI-disease","BC5CDR","BioNLP-2011-GE","BioNLP-2013-GRO"
        ],
         'TXTCLASS': ['MedDialog'],
         'RE': ['AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE'],
         'EE': ['MLEE-EE'],
}
bio = ['NCBI-disease', 'BC5CDR']
cls = ['MedDialog']
entity = [
          'BioNLP-2011-GE',
          'BioNLP-2013-GRO',
          'AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE',
          'MLEE-EE'
          ]


def mse_score(targets, preds):
    def extract_integers_from_string(s):
        integers = re.findall(r'\d+', s)
        integers = [int(num) for num in integers]
        return list(set(integers))

    ts = []
    ps = []
    for t, p in zip(targets, preds):
        t_numbers = extract_integers_from_string(t)
        p_numbers = extract_integers_from_string(p)
        if len(t_numbers) != 1 or len(p_numbers) != 1:
            t_num = 0
            p_num = 5
        elif t_numbers[0] not in [0, 1, 2, 3, 4, 5] or p_numbers[0] not in [0, 1, 2, 3, 4, 5]:
            t_num = 0
            p_num = 5
        else:
            t_num = t_numbers[0]
            p_num = p_numbers[0]
        ts.append(t_num)
        ps.append(p_num)
    n = len(ts)
    mse = sum((x - y) ** 2 for x, y in zip(ts, ps)) / n
    return mse


def post_bio(target, pred):
    def extract_tags(input_string):
        pattern = r'\[B\]|\[I\]|\[O\]'
        matches = re.findall(pattern, input_string)
        return matches

    perd_labels = extract_tags(pred)
    target_labels = extract_tags(target)
    return target_labels, perd_labels


# 为post_entity添加filter_entities参数，支持过滤指定实体
def post_entity(target, pred, filter_entities=None):
    def extract_entities_with_stack(s):
        stack = []
        entities = []
        current_entity = []
        for char in s:
            if char == '[':
                if stack:
                    current_entity.append(char)
                stack.append(char)
            elif char == ']':
                if not stack:
                    current_entity = []
                    continue
                stack.pop()
                if stack:
                    current_entity.append(char)
                else:
                    entities.append(normalize_answer(''.join(current_entity)))
                    current_entity = []
            elif stack:
                current_entity.append(char)
        return entities

    target_entities = extract_entities_with_stack(target)
    pred_entities = extract_entities_with_stack(pred)

    # 过滤in_train=true的实体
    if filter_entities is not None and len(filter_entities) > 0:
        normalized_filter = [normalize_answer(ent) for ent in filter_entities]
        target_entities = [ent for ent in target_entities if ent not in normalized_filter]
        pred_entities = [ent for ent in pred_entities if ent not in normalized_filter]

    return target_entities, pred_entities


def label_level_f1(targets, preds):
    macro_f1 = f1_score(targets, preds, labels=sorted(set(targets)), average='macro')
    recall = recall_score(targets, preds, labels=sorted(set(targets)), average='macro')
    precision = precision_score(targets, preds, labels=sorted(set(targets)), average='macro')
    return precision * 100, recall * 100, macro_f1 * 100


def entity_level_precision_recall(target, pred):
    true_counter = Counter(target)
    pred_counter = Counter(pred)

    tp = sum((true_counter & pred_counter).values())
    fp = sum((pred_counter - true_counter).values())
    fn = sum((true_counter - pred_counter).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision * 100, recall * 100, f1 * 100


def rouge(prediction: str, ground_truth: str, rouge_metric):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_aggregator': False, 'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0]


def word_level_f1(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def word_level_precision_recall(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0

    return precision * 100, recall * 100


def exact_match_score(prediction: str, ground_truth: str):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# my_evaluate添加entity_filter_lists参数，支持过滤
def my_evaluate(targets: List[str], predictions: List[str], evaluation_types: List[str],
                rouge_metric, bio_in_train_flags: List[bool] = None, entity_filter_lists: List[List[str]] = None) -> Dict:
    """
    评估函数
    Args:
        targets: 目标标签列表
        predictions: 预测结果列表
        evaluation_types: 评估类型列表
        rouge_metric: Rouge评估器
        bio_in_train_flags: 对于bio类型数据集，表示每个样本的entity_in_train标记（过滤）
        entity_filter_lists: 对于BioNLP-2011-GE/GRO，表示每个样本需要过滤的实体列表（过滤）
    """
    assert len(predictions) == len(targets), \
        f"The pred file does not have the same length as the gold data: {len(targets)} vs {len(predictions)}"

    metrics = {}

    # 过滤（NCBI-disease/BC5CDR）
    filtered_targets = targets
    filtered_predictions = predictions
    filtered_entity_filter_lists = entity_filter_lists
    n_samples = len(predictions)

    if bio_in_train_flags is not None:
        filtered_targets = []
        filtered_predictions = []
        filtered_entity_filter_lists = [] if entity_filter_lists is not None else None
        for i, (t, p) in enumerate(zip(targets, predictions)):
            if not bio_in_train_flags[i]:
                filtered_targets.append(t)
                filtered_predictions.append(p)
                if entity_filter_lists is not None:
                    filtered_entity_filter_lists.append(entity_filter_lists[i])

        n_samples = len(filtered_targets)
        if n_samples == 0:
            print("警告: 没有找到entity_in_train为false的样本，使用所有样本计算")
            filtered_targets = targets
            filtered_predictions = predictions
            filtered_entity_filter_lists = entity_filter_lists
            n_samples = len(targets)

    # 初始化新增指标
    metrics['word_precision'] = 0
    metrics['word_recall'] = 0
    metrics['entity_precision'] = 0
    metrics['entity_recall'] = 0

    for idx, (gold, pred) in datasets.tqdm(enumerate(zip(filtered_targets, filtered_predictions))):
        # 获取当前样本需要过滤的实体列表
        current_filter_entities = None
        if entity_filter_lists is not None and idx < len(filtered_entity_filter_lists):
            current_filter_entities = filtered_entity_filter_lists[idx]

        # Rouge指标
        if 'rouge' not in metrics:
            metrics['rouge'] = 0
        metrics['rouge'] += rouge(pred, gold, rouge_metric)

        # 词级F1 + 新增词级Precision/Recall
        word_f1 = word_level_f1(pred, gold)
        word_p, word_r = word_level_precision_recall(pred, gold)
        if 'f1' not in metrics:
            metrics['f1'] = 0
        metrics['f1'] += word_f1
        metrics['word_precision'] += word_p
        metrics['word_recall'] += word_r

        # 实体级指标（传入过滤列表）
        if 'entity' in evaluation_types:
            ts, ps = post_entity(gold, pred, current_filter_entities)
            entity_p, entity_r, entity_f1 = entity_level_precision_recall(ts, ps)
            metrics['entity_level_f1'] = metrics.get('entity_level_f1', 0) + entity_f1
            metrics['entity_precision'] += entity_p
            metrics['entity_recall'] += entity_r

        # 多分类任务
        if 'multicls' in evaluation_types:
            ts = gold.split(', ')
            ps = pred.split(', ')
            ts = [t.lower().strip() for t in ts]
            ps = [p.lower().strip() for p in ps]
            entity_p, entity_r, entity_f1 = entity_level_precision_recall(ts, ps)
            metrics['entity_level_f1'] = metrics.get('entity_level_f1', 0) + entity_f1
            metrics['entity_precision'] += entity_p
            metrics['entity_recall'] += entity_r

        # 精确匹配
        if 'em' in evaluation_types:
            if 'em' not in metrics:
                metrics['em'] = 0
            metrics['em'] += exact_match_score(pred, gold)

    # 归一化所有指标（按样本数平均）
    for key in metrics.keys():
        metrics[key] /= n_samples

    # Bio标签级F1（过滤的bio任务）
    if 'bio' in evaluation_types:
        ts_all = []
        ps_all = []

        for i, (t, p) in enumerate(zip(filtered_targets, filtered_predictions)):
            post_t, post_p = post_bio(t, p)
            if len(post_t) > len(post_p):
                post_p = post_p + ['N' for _ in range(len(post_t) - len(post_p))]
            else:
                post_p = post_p[:len(post_t)]

            ts_all.extend(post_t)
            ps_all.extend(post_p)

        if len(ts_all) > 0 and len(ps_all) > 0:
            label_p_all, label_r_all, label_f_all = label_level_f1(ts_all, ps_all)
            metrics['label_leval_f1'] = label_f_all
            metrics['label_leval_r'] = label_r_all
            metrics['label_leval_p'] = label_p_all
        else:
            metrics['label_leval_f1'] = 0.0
            metrics['label_leval_r'] = 0.0
            metrics['label_leval_p'] = 0.0

        # 记录使用的样本数和非重叠样本比例
        if bio_in_train_flags is not None:
            n_filtered_samples = len(filtered_targets)
            n_total_samples = len(targets)
            metrics['non_overlap_samples_used'] = n_filtered_samples
            metrics['total_samples'] = n_total_samples
            metrics['non_overlap_ratio'] = (n_filtered_samples / n_total_samples * 100) if n_total_samples > 0 else 0

    # 为BioNLP-2011-GE/GRO添加实体过滤的统计信息
    if entity_filter_lists is not None:
        total_filtered_entities = sum(len(lst) for lst in filtered_entity_filter_lists)
        metrics['total_filtered_entities'] = total_filtered_entities
        metrics['avg_filtered_entities_per_sample'] = total_filtered_entities / n_samples if n_samples > 0 else 0

    # 分类任务标签级F1
    if "cls" in evaluation_types:
        ts = [normalize_answer(t) for t in filtered_targets]
        ps = [normalize_answer(p) for p in filtered_predictions]
        label_p, label_r, label_f = label_level_f1(ts, ps)
        metrics['label_leval_f1'] = label_f
        metrics['label_leval_r'] = label_r
        metrics['label_leval_p'] = label_p

    # MSE指标
    if 'mse' in evaluation_types:
        metrics['mse'] = mse_score(filtered_targets, filtered_predictions)

    return metrics

# 扩展load_bio_original_data的file_map，支持新任务
def load_bio_original_data(test_name: str, original_data_dir: str = '.'):
    """加载bio类型数据集的原始数据文件"""
    file_map = {
        'NCBI-disease': 'NCBI-disease_test.json',
        'BC5CDR': 'BC5CDR_test.json',
        'BioNLP-2011-GE': 'BioNLP-2011-GE_test.json',
        'BioNLP-2013-GRO': 'BioNLP-2013-GRO_test.json'
    }

    if test_name not in file_map:
        return None

    file_path = os.path.join(original_data_dir, file_map[test_name])
    if not os.path.exists(file_path):
        print(f"警告: 找不到原始数据文件: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # 提取entity_in_train字段
        entity_in_train_flags = [item.get('entity_in_train', False) for item in original_data]

        # 统计信息
        n_total = len(entity_in_train_flags)
        n_false = sum(1 for flag in entity_in_train_flags if not flag)
        print(
            f"数据集 {test_name}: 总样本数={n_total}, entity_in_train=false的样本数={n_false} ({(n_false / n_total * 100):.1f}%)")

        return entity_in_train_flags
    except Exception as e:
        print(f"加载原始数据文件时出错 {file_path}: {e}")
        return None

# 新增load_entity_filter_list函数，提取需要过滤的实体列表
def load_entity_filter_list(test_name: str, original_data_dir: str = '.'):
    file_map = {
        'BioNLP-2011-GE': 'BioNLP-2011-GE_test.json',
        'BioNLP-2013-GRO': 'BioNLP-2013-GRO_test.json'
    }

    if test_name not in file_map:
        return None

    file_path = os.path.join(original_data_dir, file_map[test_name])
    if not os.path.exists(file_path):
        print(f"警告: 找不到原始数据文件: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        filter_lists = []
        total_entities = 0
        filtered_entities = 0

        for item in original_data:
            entities_detail = item.get('entities_detail', [])
            # 提取in_train=true的实体字符串
            filter_entities = []
            for ent_detail in entities_detail:
                total_entities += 1
                if ent_detail.get('in_train', False):
                    filter_entities.append(ent_detail['entity'])
                    filtered_entities += 1
            filter_lists.append(filter_entities)

        # 打印统计信息
        print(
            f"数据集 {test_name}: 总实体数={total_entities}, 需要过滤的实体数={filtered_entities} ({(filtered_entities / total_entities * 100):.1f}%)")

        return filter_lists
    except Exception as e:
        print(f"加载实体过滤列表时出错 {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="已保存的预测结果文件名（不含.json后缀）")
    parser.add_argument("--dir", type=str, required=True, help="预测结果文件所在目录")
    parser.add_argument("--original_data_dir", type=str, default=".", help="原始数据文件目录")
    args = parser.parse_args()

    name = args.name
    dir = args.dir
    original_data_dir = args.original_data_dir

    # 加载Rouge指标
    rouge_metric = evaluate.load('rouge_score', experiment_id=str(random.randint(1, 888888)))

    # 读取已保存的预测结果文件
    saved_results_path = os.path.join(dir, f"{name}.json")
    if not os.path.exists(saved_results_path):
        raise FileNotFoundError(f"未找到保存的预测结果文件：{saved_results_path}")

    with open(saved_results_path, 'r', encoding='utf-8') as f:
        saved_results = json.load(f)

    # 重新计算指标
    updated_results = {}
    for cat, test_names in tests.items():
        updated_results[cat] = {}
        for test in test_names:
            if cat not in saved_results or test not in saved_results[cat]:
                print(f"警告：{cat}->{test} 未在保存文件中找到，跳过")
                continue

            print(f'------------------------ {cat}: {test} ------------------------')
            updated_results[cat][test] = {}

            # 从保存文件中提取targets和predictions
            generated_data = saved_results[cat][test]['generated']
            targets = [item['target'] for item in generated_data]
            predictions = [item['prediction'] for item in generated_data]

            # 确定数据集类型
            types = []
            if test in cls:
                types.append('cls')

            elif test in entity:
                types.append('entity')

            elif test in bio:
                types.append('bio')

            # 加载过滤标志（NCBI-disease/BC5CDR）
            bio_in_train_flags = None
            if test in bio:
                print(f"正在为{test}加载原始数据文件...")
                bio_in_train_flags = load_bio_original_data(test, original_data_dir)
                if bio_in_train_flags is not None and len(bio_in_train_flags) != len(targets):
                    bio_in_train_flags = None

            #加载过滤列表（BioNLP-2011-GE/BioNLP-2013-GRO）
            entity_filter_lists = None
            if test in ['BioNLP-2011-GE', 'BioNLP-2013-GRO']:
                print(f"正在为{test}加载实体过滤列表...")
                entity_filter_lists = load_entity_filter_list(test, original_data_dir)
                if entity_filter_lists is not None and len(entity_filter_lists) != len(targets):
                    entity_filter_lists = None

            # 重新计算指标（传入实体过滤列表）
            updated_metrics = my_evaluate(targets, predictions, types, rouge_metric, bio_in_train_flags, entity_filter_lists)
            updated_results[cat][test]['generated'] = generated_data
            updated_results[cat][test]['metrics'] = updated_metrics

            clean_metrics = {}
            if test in bio or test in cls:
                clean_metrics = {
                    "Label-P": round(updated_metrics.get("label_leval_p", 0.0), 2),
                    "Label-R": round(updated_metrics.get("label_leval_r", 0.0), 2),
                    "Label-F1": round(updated_metrics.get("label_leval_f1", 0.0), 2)
                }
                print(f"Label-P: {clean_metrics['Label-P']:.2f}")
                print(f"Label-R: {clean_metrics['Label-R']:.2f}")
                print(f"Label-F1: {clean_metrics['Label-F1']:.2f}")
            elif test in entity:
                clean_metrics = {
                    "Entity-P": round(updated_metrics.get("entity_precision", 0.0), 2),
                    "Entity-R": round(updated_metrics.get("entity_recall", 0.0), 2),
                    "Entity-F1": round(updated_metrics.get("entity_level_f1", 0.0), 2)
                }
                print(f"Entity-P: {clean_metrics['Entity-P']:.2f}")
                print(f"Entity-R: {clean_metrics['Entity-R']:.2f}")
                print(f"Entity-F1: {clean_metrics['Entity-F1']:.2f}")

            updated_results[cat][test]['metrics'] = clean_metrics


    # 保存更新后的结果
    output_path = os.path.join(dir, f"{name}_with_pr_bio_filtered.json")
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(updated_results, out, ensure_ascii=False, indent=4)
    print(f"\n更新后的结果已保存至：{output_path}")


if __name__ == '__main__':
    main()