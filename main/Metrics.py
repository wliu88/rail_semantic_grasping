# This file contains score computations for all experiments

import numpy as np
import pandas as pd
from collections import OrderedDict

import main.DataSpecification as DataSpecification


def compute_aps(obj_gts, obj_preds):
    pos_ap = None
    neg_ap = None
    hit_1 = 0
    hit_2 = 0
    hit_3 = 0
    hits_5 = 0

    # compute ranking score for positive preference
    if 1 in obj_gts:
        pos_preds = obj_preds[:, -1]
        sort_indices = np.argsort(pos_preds)[::-1]
        ranked_gts = obj_gts[sort_indices]
        print("pos:\n", ranked_gts)

        num_corrects = 0.0
        num_corrects_5 = 0.0
        num_predictions = 0.0
        total_precisions = []
        for i in range(len(ranked_gts)):
            num_predictions += 1
            if ranked_gts[i] == 1:
                if i == 0:
                    hit_1 = 1
                if i == 1 and hit_1 == 1:
                    hit_2 = 1
                if i == 2 and hit_2 == 1:
                    hit_3 = 1
                if i < 5:
                    num_corrects_5 += 1
                num_corrects += 1
                total_precisions.append(num_corrects / num_predictions)
        pos_ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None
        hits_5 = num_corrects_5 / 5

    # compute ranking score for negative preference
    if -1 in obj_gts:
        neg_preds = obj_preds[:, 0]
        sort_indices = np.argsort(neg_preds)[::-1]
        ranked_gts = obj_gts[sort_indices]
        # print("neg:\n", ranked_gts)

        num_corrects = 0.0
        num_predictions = 0.0
        total_precisions = []
        for i in range(len(ranked_gts)):
            num_predictions += 1
            if ranked_gts[i] == -1:
                num_corrects += 1
                total_precisions.append(num_corrects / num_predictions)
        neg_ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None

    return pos_ap, neg_ap, hit_1, hit_2, hit_3, hits_5


def score_1(results):
    # this is used to group raw scores based on object classes and tasks
    raw_scores = OrderedDict()
    for task in DataSpecification.TASKS:
        raw_scores[task] = OrderedDict()
        for obj in DataSpecification.OBJECTS:
            # first stores ap scores for positive preference, second for negative preference, third for number of
            # testing examples
            raw_scores[task][obj] = [[], [], 0]

    for result in results:
        description, gts, preds = result
        task, object_class, iter = description.split(":")

        raw_scores[task][object_class][2] = len(gts)

        for obj_preds, obj_gts in zip(preds, gts):
            obj_preds, obj_gts = np.array(obj_preds), np.array(obj_gts)

            pos_ap, neg_ap, hit_1, hit_2, hit_3, hits_5 = compute_aps(obj_gts, obj_preds)
            if pos_ap is not None:
                raw_scores[task][object_class][0].append(pos_ap)
            if neg_ap is not None:
                raw_scores[task][object_class][1].append(neg_ap)

    # print(raw_scores)

    # format scores for better visualization
    pos_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    neg_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    num_examples = np.zeros([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)])
    for ti, task in enumerate(raw_scores):
        for oi, object_class in enumerate(raw_scores[task]):
            pos_map[oi][ti] = np.average(raw_scores[task][object_class][0])
            neg_map[oi][ti] = np.average(raw_scores[task][object_class][1])
            num_examples[oi][ti] = raw_scores[task][object_class][2]

    pos_map_avg = np.nanmean(pos_map)
    neg_map_avg = np.nanmean(neg_map)

    pos_map_pd = pd.DataFrame(pos_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)
    neg_map_pd = pd.DataFrame(neg_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)

    # remove nans to compute weighted avg
    print("Pos ap raw: {}".format(list(pos_map[~np.isnan(pos_map)].flat)))
    print("Neg ap raw: {}".format(list(neg_map[~np.isnan(neg_map)].flat)))

    # remove nans to compute weighted avg
    pos_map[np.isnan(pos_map)] = 0
    neg_map[np.isnan(neg_map)] = 0
    pos_map_weighted_avg = np.average(pos_map, weights=num_examples)
    neg_map_weighted_avg = np.average(neg_map, weights=num_examples)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Positive preference MAP:\n", pos_map_pd)
        print("Mean: {}, Weighted Average: {}".format(pos_map_avg, pos_map_weighted_avg))
        print("\nNegative preference MAP:\n", neg_map_pd)
        print("Mean: {}, Weighted Average: {}".format(neg_map_avg, neg_map_weighted_avg))


def score_2(results):
    # this is used to group raw scores based on object classes and tasks
    raw_scores = OrderedDict()
    for task in DataSpecification.TASKS:
        raw_scores[task] = OrderedDict()
        for obj in DataSpecification.OBJECTS:
            # first stores ap score for positive preference, second for negative preference, third for number of
            # testing examples
            raw_scores[task][obj] = [[], [], 0]

    for result in results:
        description, gts, preds = result
        task, object_class = description.split(":")

        # the number of object instances
        raw_scores[task][object_class][2] = len(gts)

        for obj_preds, obj_gts in zip(preds, gts):
            obj_preds, obj_gts = np.array(obj_preds), np.array(obj_gts)

            pos_ap, neg_ap, hit_1, hit_2, hit_3, hits_5 = compute_aps(obj_gts, obj_preds)
            if pos_ap is not None:
                raw_scores[task][object_class][0].append(pos_ap)
            if neg_ap is not None:
                raw_scores[task][object_class][1].append(neg_ap)

    # format scores for better visualization
    pos_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    neg_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    num_examples = np.zeros([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)])
    for ti, task in enumerate(raw_scores):
        for oi, object_class in enumerate(raw_scores[task]):
            pos_map[oi][ti] = np.average(raw_scores[task][object_class][0])
            neg_map[oi][ti] = np.average(raw_scores[task][object_class][1])
            num_examples[oi][ti] = raw_scores[task][object_class][2]

    pos_map_avg = np.nanmean(pos_map)
    neg_map_avg = np.nanmean(neg_map)

    pos_map_pd = pd.DataFrame(pos_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)
    neg_map_pd = pd.DataFrame(neg_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)
    pos_map_pd.loc['mean'] = pos_map_pd.mean()
    neg_map_pd.loc['mean'] = neg_map_pd.mean()

    # remove nans to compute weighted avg
    print("Pos ap raw: {}".format(list(pos_map[~np.isnan(pos_map)].flat)))
    print("Neg ap raw: {}".format(list(neg_map[~np.isnan(neg_map)].flat)))

    pos_map[np.isnan(pos_map)] = 0
    neg_map[np.isnan(neg_map)] = 0
    pos_map_weighted_avg = np.average(pos_map, weights=num_examples)
    neg_map_weighted_avg = np.average(neg_map, weights=num_examples)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Positive preference MAP:\n", pos_map_pd)
        print("Mean: {}, Weighted Average: {}".format(pos_map_avg, pos_map_weighted_avg))
        print("\nNegative preference MAP:\n", neg_map_pd)
        print("Mean: {}, Weighted Average: {}".format(neg_map_avg, neg_map_weighted_avg))
        print("\nNumber of object instances:\n", num_examples)


def score_3(results):
    # this is used to group raw scores based on object classes and tasks
    raw_scores = OrderedDict()
    for task in DataSpecification.TASKS:
        raw_scores[task] = OrderedDict()
        for obj in DataSpecification.OBJECTS:
            # first stores ap scores for positive preference, second for negative preference, third for number of
            # testing examples
            raw_scores[task][obj] = [[], [], 0]

    for result in results:
        description, gts, preds = result
        task, object_class = description.split(":")

        raw_scores[task][object_class][2] = len(gts)

        for obj_preds, obj_gts in zip(preds, gts):
            obj_preds, obj_gts = np.array(obj_preds), np.array(obj_gts)

            pos_ap, neg_ap, hit_1, hit_2, hit_3, hits_5 = compute_aps(obj_gts, obj_preds)
            if pos_ap is not None:
                raw_scores[task][object_class][0].append(pos_ap)
            if neg_ap is not None:
                raw_scores[task][object_class][1].append(neg_ap)

    print(raw_scores)

    # format scores for better visualization
    pos_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    neg_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    num_examples = np.zeros([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)])
    for ti, task in enumerate(raw_scores):
        for oi, object_class in enumerate(raw_scores[task]):
            pos_map[oi][ti] = np.average(raw_scores[task][object_class][0])
            neg_map[oi][ti] = np.average(raw_scores[task][object_class][1])
            num_examples[oi][ti] = raw_scores[task][object_class][2]

    pos_map_avg = np.nanmean(pos_map)
    neg_map_avg = np.nanmean(neg_map)

    pos_map_pd = pd.DataFrame(pos_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)
    neg_map_pd = pd.DataFrame(neg_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)

    # remove nans to compute weighted avg
    pos_map[np.isnan(pos_map)] = 0
    neg_map[np.isnan(neg_map)] = 0
    pos_map_weighted_avg = np.average(pos_map, weights=num_examples)
    neg_map_weighted_avg = np.average(neg_map, weights=num_examples)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Positive preference MAP:\n", pos_map_pd)
        print("Mean: {}, Weighted Average: {}".format(pos_map_avg, pos_map_weighted_avg))
        print("\nNegative preference MAP:\n", neg_map_pd)
        print("Mean: {}, Weighted Average: {}".format(neg_map_avg, neg_map_weighted_avg))


def score_4(results):
    # this is used to group raw scores based on positive and negative preferences
    raw_pos_scores = []
    raw_neg_scores = []
    hit_1_scores = []
    hit_2_scores = []
    hit_3_scores = []
    hits_5_scores = []

    for result in results:
        description, gts, preds = result
        test_iter = description.split(":")

        pos_aps = []
        neg_aps = []
        hit_1s = []
        hit_2s = []
        hit_3s = []
        hits_5s = []

        for obj_preds, obj_gts in zip(preds, gts):
            obj_preds, obj_gts = np.array(obj_preds), np.array(obj_gts)

            pos_ap, neg_ap, hit_1, hit_2, hit_3, hits_5 = compute_aps(obj_gts, obj_preds)
            if pos_ap is not None:
                pos_aps.append(pos_ap)
            if neg_ap is not None:
                neg_aps.append(neg_ap)
            hit_1s.append(hit_1)
            hit_2s.append(hit_2)
            hit_3s.append(hit_3)
            hits_5s.append(hits_5)

        raw_pos_scores.append(sum(pos_aps) * 1.0/ len(pos_aps))
        raw_neg_scores.append(sum(neg_aps) * 1.0 / len(neg_aps))
        hit_1_scores.append(sum(hit_1s) * 1.0 / len(hit_1s))
        hit_2_scores.append(sum(hit_2s) * 1.0 / len(hit_2s))
        hit_3_scores.append(sum(hit_3s) * 1.0 / len(hit_3s))
        hits_5_scores.append(sum(hits_5s) * 1.0 / len(hits_5s))

    print("Pos ap raw: {}".format(raw_pos_scores))
    print("Neg ap raw: {}".format(raw_neg_scores))
    print("Hit@1 raw: {}".format(hit_1_scores))
    print("Hit@2 raw: {}".format(hit_2_scores))
    print("Hit@3 raw: {}".format(hit_3_scores))
    print("Hits@5 raw: {}".format(hits_5_scores))

    # format scores for better visualization
    pos_map = np.average(raw_pos_scores)
    neg_map = np.average(raw_neg_scores)

    print("\nPositive preference MAP:", pos_map)
    print("Negative preference MAP:", neg_map)
    print("Hit@1: {}".format(np.average(hit_1_scores)))
    print("Hit@2: {}".format(np.average(hit_2_scores)))
    print("Hit@3: {}".format(np.average(hit_3_scores)))
    print("Hits@5: {}".format(np.average(hits_5_scores)))

def score_embedding_3(results):
    # this is used to group raw scores based on object classes and tasks
    raw_scores = OrderedDict()
    for task in DataSpecification.TASKS:
        raw_scores[task] = OrderedDict()
        for obj in DataSpecification.OBJECTS:
            # first stores ap scores for positive preference, second for negative preference, third for number of
            # testing examples
            raw_scores[task][obj] = [[], [], 0]

    for result in results:
        description, gts, preds = result
        task, object_class = description.split(":")

        raw_scores[task][object_class][2] = len(gts)

        for obj_preds, obj_gts in zip(preds, gts):
            obj_preds, obj_gts = np.array(obj_preds), np.array(obj_gts)

            # compute ranking score for positive preference
            pos_ap = None
            if 1 in obj_gts:
                pos_preds = obj_preds
                sort_indices = np.argsort(pos_preds) # [::-1]
                ranked_gts = obj_gts[sort_indices]
                print("pos:\n", ranked_gts)

                num_corrects = 0.0
                num_predictions = 0.0
                total_precisions = []
                for i in range(len(ranked_gts)):
                    num_predictions += 1
                    if ranked_gts[i] == 1:
                        num_corrects += 1
                        total_precisions.append(num_corrects / num_predictions)
                pos_ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None

            if pos_ap is not None:
                raw_scores[task][object_class][0].append(pos_ap)

    print(raw_scores)

    # format scores for better visualization
    pos_map = np.full([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)], np.nan)
    num_examples = np.zeros([len(DataSpecification.OBJECTS), len(DataSpecification.TASKS)])
    for ti, task in enumerate(raw_scores):
        for oi, object_class in enumerate(raw_scores[task]):
            pos_map[oi][ti] = np.average(raw_scores[task][object_class][0])
            num_examples[oi][ti] = raw_scores[task][object_class][2]

    pos_map_avg = np.nanmean(pos_map)

    pos_map_pd = pd.DataFrame(pos_map, index=DataSpecification.OBJECTS, columns=DataSpecification.TASKS)

    # remove nans to compute weighted avg
    pos_map[np.isnan(pos_map)] = 0
    pos_map_weighted_avg = np.average(pos_map, weights=num_examples)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Positive preference MAP:\n", pos_map_pd)
        print("Mean: {}, Weighted Average: {}".format(pos_map_avg, pos_map_weighted_avg))
