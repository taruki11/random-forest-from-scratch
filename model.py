"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

# ----------------- Step 1: 不纯度计算 (Gini Impurity) -----------------
def impurity(labels):
    """计算一维整型类别标签的基尼不纯度（0 为纯净，越混合值越大）。"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))

# ----------------- Step 2: 划分数据集 -----------------
def split_dataset(features, labels, feature_index, threshold):
    """根据指定特征与阈值将样本切分为左右两组 (左 <= threshold, 右 > threshold)。"""
    mask = features[:, feature_index] <= threshold
    return features[mask], labels[mask], features[~mask], labels[~mask]

# ----------------- Step 3: 分裂质量评分 (加权不纯度降低量) -----------------
def split_score(parent_labels, left_labels, right_labels):
    """计算分裂带来的不纯度减少量（信息增益），值越大表示分裂后越纯。"""
    n = len(parent_labels)
    if n == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_imp = impurity(parent_labels)
    l_imp = impurity(left_labels)
    r_imp = impurity(right_labels)
    weighted_child_imp = (len(left_labels) / n) * l_imp + (len(right_labels) / n) * r_imp
    return float(p_imp - weighted_child_imp)

# ----------------- Step 4: 寻找最优划分 -----------------
def best_split(features, labels, feature_indices):
    """在给定的候选特征子集中寻找能够最大化纯度提升的 (特征索引, 阈值)。"""
    best_score = 0.0
    best_feat = None
    best_thresh = None

    for f_idx in feature_indices:
        col_vals = features[:, f_idx]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 1:
            continue
        # 取相邻有序值的平均值作为候选切分点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for th in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, th)
            score = split_score(labels, left_y, right_y)
            if score > best_score:
                best_score = score
                best_feat = f_idx
                best_thresh = th

    if best_score <= 1e-9 or best_feat is None:
        return {'feature_index': None, 'threshold': None, 'score': 0.0}
    return {
        'feature_index': int(best_feat),
        'threshold': float(best_thresh),
        'score': float(best_score)
    }

# ----------------- Step 5: 停止分裂判定 -----------------
def should_stop(labels, depth, max_depth, min_samples_split):
    """判定当前节点是否应当停止生长转为叶子节点。"""
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return True
    if depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# ----------------- Step 6: 叶子节点类别预测 -----------------
def leaf_prediction(labels):
    """返回叶子节点样本集中出现频次最高的多数类别标签。"""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])

# ----------------- Step 7: 递归构建单棵决策树 -----------------
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """递归构建决策树，返回嵌套字典结构的节点。"""
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    d = features.shape[1]
    cand_indices = list(range(d)) if feature_subset is None else feature_subset
    split = best_split(features, labels, cand_indices)
    
    if split['score'] <= 0.0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    feat_idx = split['feature_index']
    thresh = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feat_idx, thresh)
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    left_child = build_tree(left_X, left_y, max_depth=max_depth, min_samples_split=min_samples_split,
                            feature_subset=feature_subset, depth=depth + 1)
    right_child = build_tree(right_X, right_y, max_depth=max_depth, min_samples_split=min_samples_split,
                             feature_subset=feature_subset, depth=depth + 1)

    return {
        'leaf': False,
        'feature_index': int(feat_idx),
        'threshold': float(thresh),
        'left': left_child,
        'right': right_child
    }

# ----------------- Step 8: 单个样本决策树推理 -----------------
def predict_example_tree(tree, example):
    """沿决策树向下路由单个样本直到叶子节点，返回预测值。"""
    curr = tree
    while not curr['leaf']:
        if example[curr['feature_index']] <= curr['threshold']:
            curr = curr['left']
        else:
            curr = curr['right']
    return int(curr['prediction'])

# ----------------- Step 9: 批量数据决策树推理 -----------------
def predict_tree(tree, features):
    """使用单棵决策树对整个特征矩阵的所有样本进行批量预测。"""
    return np.array([predict_example_tree(tree, row) for row in features], dtype=int)

# ----------------- Step 10: 自助采样 (Bootstrap Sampling) -----------------
def bootstrap_sample(features, labels, rng):
    """有放回随机抽取与原样本量相同数量的行样本。"""
    n = len(features)
    indices = rng.choice(n, size=n, replace=True)
    return features[indices], labels[indices]

# ----------------- Step 11: 特征子集随机选取 -----------------
def feature_subset(num_features, num_to_pick, rng):
    """无放回随机抽取 num_to_pick 个不重复特征列索引。"""
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ----------------- Step 12: 随机森林训练 -----------------
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """训练随机森林：对每棵树执行行 Bootstrap 与列特征抽样并拟合。"""
    rng = np.random.default_rng(random_state)
    d = features.shape[1]
    k = max(1, int(round(np.sqrt(d)))) if num_features_per_split is None else num_features_per_split

    forest = []
    for _ in range(num_trees):
        boot_X, boot_y = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, k, rng)
        tree = build_tree(boot_X, boot_y, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feat_indices)
        forest.append({'tree': tree, 'feature_indices': feat_indices})
    return forest

# ----------------- Step 13: 多数投票聚合预测 -----------------
def combine_predictions(tree_predictions):
    """对森林中所有树的预测结果按样本列进行多数投票（Majority Vote）。"""
    tp = np.asarray(tree_predictions)
    n = tp.shape[1]
    out = np.zeros(n, dtype=int)
    for j in range(n):
        col = tp[:, j]
        vals, counts = np.unique(col, return_counts=True)
        out[j] = vals[np.argmax(counts)]
    return out

# ----------------- Step 14: 随机森林批量预测 -----------------
def predict_forest(forest, features):
    """使用训练完成的随机森林对特征数据集进行联合预测。"""
    tp = np.array([predict_tree(t['tree'], features) for t in forest])
    return combine_predictions(tp)

# ----------------- Step 15: 准确率评估 -----------------
def accuracy(predictions, labels):
    """计算预测标签与真实标签的匹配比率。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float(np.mean(predictions == labels))

# Step 2 - split_dataset
import numpy as np

# ----------------- Step 1: 不纯度计算 (Gini Impurity) -----------------
def impurity(labels):
    """计算一维整型类别标签的基尼不纯度（0 为纯净，越混合值越大）。"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))

# ----------------- Step 2: 划分数据集 -----------------
def split_dataset(features, labels, feature_index, threshold):
    """根据指定特征与阈值将样本切分为左右两组 (左 <= threshold, 右 > threshold)。"""
    mask = features[:, feature_index] <= threshold
    return features[mask], labels[mask], features[~mask], labels[~mask]

# ----------------- Step 3: 分裂质量评分 (加权不纯度降低量) -----------------
def split_score(parent_labels, left_labels, right_labels):
    """计算分裂带来的不纯度减少量（信息增益），值越大表示分裂后越纯。"""
    n = len(parent_labels)
    if n == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_imp = impurity(parent_labels)
    l_imp = impurity(left_labels)
    r_imp = impurity(right_labels)
    weighted_child_imp = (len(left_labels) / n) * l_imp + (len(right_labels) / n) * r_imp
    return float(p_imp - weighted_child_imp)

# ----------------- Step 4: 寻找最优划分 -----------------
def best_split(features, labels, feature_indices):
    """在给定的候选特征子集中寻找能够最大化纯度提升的 (特征索引, 阈值)。"""
    best_score = 0.0
    best_feat = None
    best_thresh = None

    for f_idx in feature_indices:
        col_vals = features[:, f_idx]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 1:
            continue
        # 取相邻有序值的平均值作为候选切分点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for th in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, th)
            score = split_score(labels, left_y, right_y)
            if score > best_score:
                best_score = score
                best_feat = f_idx
                best_thresh = th

    if best_score <= 1e-9 or best_feat is None:
        return {'feature_index': None, 'threshold': None, 'score': 0.0}
    return {
        'feature_index': int(best_feat),
        'threshold': float(best_thresh),
        'score': float(best_score)
    }

# ----------------- Step 5: 停止分裂判定 -----------------
def should_stop(labels, depth, max_depth, min_samples_split):
    """判定当前节点是否应当停止生长转为叶子节点。"""
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return True
    if depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# ----------------- Step 6: 叶子节点类别预测 -----------------
def leaf_prediction(labels):
    """返回叶子节点样本集中出现频次最高的多数类别标签。"""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])

# ----------------- Step 7: 递归构建单棵决策树 -----------------
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """递归构建决策树，返回嵌套字典结构的节点。"""
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    d = features.shape[1]
    cand_indices = list(range(d)) if feature_subset is None else feature_subset
    split = best_split(features, labels, cand_indices)
    
    if split['score'] <= 0.0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    feat_idx = split['feature_index']
    thresh = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feat_idx, thresh)
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    left_child = build_tree(left_X, left_y, max_depth=max_depth, min_samples_split=min_samples_split,
                            feature_subset=feature_subset, depth=depth + 1)
    right_child = build_tree(right_X, right_y, max_depth=max_depth, min_samples_split=min_samples_split,
                             feature_subset=feature_subset, depth=depth + 1)

    return {
        'leaf': False,
        'feature_index': int(feat_idx),
        'threshold': float(thresh),
        'left': left_child,
        'right': right_child
    }

# ----------------- Step 8: 单个样本决策树推理 -----------------
def predict_example_tree(tree, example):
    """沿决策树向下路由单个样本直到叶子节点，返回预测值。"""
    curr = tree
    while not curr['leaf']:
        if example[curr['feature_index']] <= curr['threshold']:
            curr = curr['left']
        else:
            curr = curr['right']
    return int(curr['prediction'])

# ----------------- Step 9: 批量数据决策树推理 -----------------
def predict_tree(tree, features):
    """使用单棵决策树对整个特征矩阵的所有样本进行批量预测。"""
    return np.array([predict_example_tree(tree, row) for row in features], dtype=int)

# ----------------- Step 10: 自助采样 (Bootstrap Sampling) -----------------
def bootstrap_sample(features, labels, rng):
    """有放回随机抽取与原样本量相同数量的行样本。"""
    n = len(features)
    indices = rng.choice(n, size=n, replace=True)
    return features[indices], labels[indices]

# ----------------- Step 11: 特征子集随机选取 -----------------
def feature_subset(num_features, num_to_pick, rng):
    """无放回随机抽取 num_to_pick 个不重复特征列索引。"""
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ----------------- Step 12: 随机森林训练 -----------------
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """训练随机森林：对每棵树执行行 Bootstrap 与列特征抽样并拟合。"""
    rng = np.random.default_rng(random_state)
    d = features.shape[1]
    k = max(1, int(round(np.sqrt(d)))) if num_features_per_split is None else num_features_per_split

    forest = []
    for _ in range(num_trees):
        boot_X, boot_y = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, k, rng)
        tree = build_tree(boot_X, boot_y, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feat_indices)
        forest.append({'tree': tree, 'feature_indices': feat_indices})
    return forest

# ----------------- Step 13: 多数投票聚合预测 -----------------
def combine_predictions(tree_predictions):
    """对森林中所有树的预测结果按样本列进行多数投票（Majority Vote）。"""
    tp = np.asarray(tree_predictions)
    n = tp.shape[1]
    out = np.zeros(n, dtype=int)
    for j in range(n):
        col = tp[:, j]
        vals, counts = np.unique(col, return_counts=True)
        out[j] = vals[np.argmax(counts)]
    return out

# ----------------- Step 14: 随机森林批量预测 -----------------
def predict_forest(forest, features):
    """使用训练完成的随机森林对特征数据集进行联合预测。"""
    tp = np.array([predict_tree(t['tree'], features) for t in forest])
    return combine_predictions(tp)

# ----------------- Step 15: 准确率评估 -----------------
def accuracy(predictions, labels):
    """计算预测标签与真实标签的匹配比率。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float(np.mean(predictions == labels))

# Step 3 - split_score
import numpy as np

# ----------------- Step 1: 不纯度计算 (Gini Impurity) -----------------
def impurity(labels):
    """计算一维整型类别标签的基尼不纯度（0 为纯净，越混合值越大）。"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))

# ----------------- Step 2: 划分数据集 -----------------
def split_dataset(features, labels, feature_index, threshold):
    """根据指定特征与阈值将样本切分为左右两组 (左 <= threshold, 右 > threshold)。"""
    mask = features[:, feature_index] <= threshold
    return features[mask], labels[mask], features[~mask], labels[~mask]

# ----------------- Step 3: 分裂质量评分 (加权不纯度降低量) -----------------
def split_score(parent_labels, left_labels, right_labels):
    """计算分裂带来的不纯度减少量（信息增益），值越大表示分裂后越纯。"""
    n = len(parent_labels)
    if n == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_imp = impurity(parent_labels)
    l_imp = impurity(left_labels)
    r_imp = impurity(right_labels)
    weighted_child_imp = (len(left_labels) / n) * l_imp + (len(right_labels) / n) * r_imp
    return float(p_imp - weighted_child_imp)

# ----------------- Step 4: 寻找最优划分 -----------------
def best_split(features, labels, feature_indices):
    """在给定的候选特征子集中寻找能够最大化纯度提升的 (特征索引, 阈值)。"""
    best_score = 0.0
    best_feat = None
    best_thresh = None

    for f_idx in feature_indices:
        col_vals = features[:, f_idx]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 1:
            continue
        # 取相邻有序值的平均值作为候选切分点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for th in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, th)
            score = split_score(labels, left_y, right_y)
            if score > best_score:
                best_score = score
                best_feat = f_idx
                best_thresh = th

    if best_score <= 1e-9 or best_feat is None:
        return {'feature_index': None, 'threshold': None, 'score': 0.0}
    return {
        'feature_index': int(best_feat),
        'threshold': float(best_thresh),
        'score': float(best_score)
    }

# ----------------- Step 5: 停止分裂判定 -----------------
def should_stop(labels, depth, max_depth, min_samples_split):
    """判定当前节点是否应当停止生长转为叶子节点。"""
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return True
    if depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# ----------------- Step 6: 叶子节点类别预测 -----------------
def leaf_prediction(labels):
    """返回叶子节点样本集中出现频次最高的多数类别标签。"""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])

# ----------------- Step 7: 递归构建单棵决策树 -----------------
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """递归构建决策树，返回嵌套字典结构的节点。"""
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    d = features.shape[1]
    cand_indices = list(range(d)) if feature_subset is None else feature_subset
    split = best_split(features, labels, cand_indices)
    
    if split['score'] <= 0.0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    feat_idx = split['feature_index']
    thresh = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feat_idx, thresh)
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    left_child = build_tree(left_X, left_y, max_depth=max_depth, min_samples_split=min_samples_split,
                            feature_subset=feature_subset, depth=depth + 1)
    right_child = build_tree(right_X, right_y, max_depth=max_depth, min_samples_split=min_samples_split,
                             feature_subset=feature_subset, depth=depth + 1)

    return {
        'leaf': False,
        'feature_index': int(feat_idx),
        'threshold': float(thresh),
        'left': left_child,
        'right': right_child
    }

# ----------------- Step 8: 单个样本决策树推理 -----------------
def predict_example_tree(tree, example):
    """沿决策树向下路由单个样本直到叶子节点，返回预测值。"""
    curr = tree
    while not curr['leaf']:
        if example[curr['feature_index']] <= curr['threshold']:
            curr = curr['left']
        else:
            curr = curr['right']
    return int(curr['prediction'])

# ----------------- Step 9: 批量数据决策树推理 -----------------
def predict_tree(tree, features):
    """使用单棵决策树对整个特征矩阵的所有样本进行批量预测。"""
    return np.array([predict_example_tree(tree, row) for row in features], dtype=int)

# ----------------- Step 10: 自助采样 (Bootstrap Sampling) -----------------
def bootstrap_sample(features, labels, rng):
    """有放回随机抽取与原样本量相同数量的行样本。"""
    n = len(features)
    indices = rng.choice(n, size=n, replace=True)
    return features[indices], labels[indices]

# ----------------- Step 11: 特征子集随机选取 -----------------
def feature_subset(num_features, num_to_pick, rng):
    """无放回随机抽取 num_to_pick 个不重复特征列索引。"""
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ----------------- Step 12: 随机森林训练 -----------------
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """训练随机森林：对每棵树执行行 Bootstrap 与列特征抽样并拟合。"""
    rng = np.random.default_rng(random_state)
    d = features.shape[1]
    k = max(1, int(round(np.sqrt(d)))) if num_features_per_split is None else num_features_per_split

    forest = []
    for _ in range(num_trees):
        boot_X, boot_y = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, k, rng)
        tree = build_tree(boot_X, boot_y, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feat_indices)
        forest.append({'tree': tree, 'feature_indices': feat_indices})
    return forest

# ----------------- Step 13: 多数投票聚合预测 -----------------
def combine_predictions(tree_predictions):
    """对森林中所有树的预测结果按样本列进行多数投票（Majority Vote）。"""
    tp = np.asarray(tree_predictions)
    n = tp.shape[1]
    out = np.zeros(n, dtype=int)
    for j in range(n):
        col = tp[:, j]
        vals, counts = np.unique(col, return_counts=True)
        out[j] = vals[np.argmax(counts)]
    return out

# ----------------- Step 14: 随机森林批量预测 -----------------
def predict_forest(forest, features):
    """使用训练完成的随机森林对特征数据集进行联合预测。"""
    tp = np.array([predict_tree(t['tree'], features) for t in forest])
    return combine_predictions(tp)

# ----------------- Step 15: 准确率评估 -----------------
def accuracy(predictions, labels):
    """计算预测标签与真实标签的匹配比率。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float(np.mean(predictions == labels))

# Step 4 - best_split
import numpy as np

# ----------------- Step 1: 不纯度计算 (Gini Impurity) -----------------
def impurity(labels):
    """计算一维整型类别标签的基尼不纯度（0 为纯净，越混合值越大）。"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))

# ----------------- Step 2: 划分数据集 -----------------
def split_dataset(features, labels, feature_index, threshold):
    """根据指定特征与阈值将样本切分为左右两组 (左 <= threshold, 右 > threshold)。"""
    mask = features[:, feature_index] <= threshold
    return features[mask], labels[mask], features[~mask], labels[~mask]

# ----------------- Step 3: 分裂质量评分 (加权不纯度降低量) -----------------
def split_score(parent_labels, left_labels, right_labels):
    """计算分裂带来的不纯度减少量（信息增益），值越大表示分裂后越纯。"""
    n = len(parent_labels)
    if n == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_imp = impurity(parent_labels)
    l_imp = impurity(left_labels)
    r_imp = impurity(right_labels)
    weighted_child_imp = (len(left_labels) / n) * l_imp + (len(right_labels) / n) * r_imp
    return float(p_imp - weighted_child_imp)

# ----------------- Step 4: 寻找最优划分 -----------------
def best_split(features, labels, feature_indices):
    """在给定的候选特征子集中寻找能够最大化纯度提升的 (特征索引, 阈值)。"""
    best_score = 0.0
    best_feat = None
    best_thresh = None

    for f_idx in feature_indices:
        col_vals = features[:, f_idx]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 1:
            continue
        # 取相邻有序值的平均值作为候选切分点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for th in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, th)
            score = split_score(labels, left_y, right_y)
            if score > best_score:
                best_score = score
                best_feat = f_idx
                best_thresh = th

    if best_score <= 1e-9 or best_feat is None:
        return {'feature_index': None, 'threshold': None, 'score': 0.0}
    return {
        'feature_index': int(best_feat),
        'threshold': float(best_thresh),
        'score': float(best_score)
    }

# ----------------- Step 5: 停止分裂判定 -----------------
def should_stop(labels, depth, max_depth, min_samples_split):
    """判定当前节点是否应当停止生长转为叶子节点。"""
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return True
    if depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# ----------------- Step 6: 叶子节点类别预测 -----------------
def leaf_prediction(labels):
    """返回叶子节点样本集中出现频次最高的多数类别标签。"""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])

# ----------------- Step 7: 递归构建单棵决策树 -----------------
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """递归构建决策树，返回嵌套字典结构的节点。"""
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    d = features.shape[1]
    cand_indices = list(range(d)) if feature_subset is None else feature_subset
    split = best_split(features, labels, cand_indices)
    
    if split['score'] <= 0.0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    feat_idx = split['feature_index']
    thresh = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feat_idx, thresh)
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    left_child = build_tree(left_X, left_y, max_depth=max_depth, min_samples_split=min_samples_split,
                            feature_subset=feature_subset, depth=depth + 1)
    right_child = build_tree(right_X, right_y, max_depth=max_depth, min_samples_split=min_samples_split,
                             feature_subset=feature_subset, depth=depth + 1)

    return {
        'leaf': False,
        'feature_index': int(feat_idx),
        'threshold': float(thresh),
        'left': left_child,
        'right': right_child
    }

# ----------------- Step 8: 单个样本决策树推理 -----------------
def predict_example_tree(tree, example):
    """沿决策树向下路由单个样本直到叶子节点，返回预测值。"""
    curr = tree
    while not curr['leaf']:
        if example[curr['feature_index']] <= curr['threshold']:
            curr = curr['left']
        else:
            curr = curr['right']
    return int(curr['prediction'])

# ----------------- Step 9: 批量数据决策树推理 -----------------
def predict_tree(tree, features):
    """使用单棵决策树对整个特征矩阵的所有样本进行批量预测。"""
    return np.array([predict_example_tree(tree, row) for row in features], dtype=int)

# ----------------- Step 10: 自助采样 (Bootstrap Sampling) -----------------
def bootstrap_sample(features, labels, rng):
    """有放回随机抽取与原样本量相同数量的行样本。"""
    n = len(features)
    indices = rng.choice(n, size=n, replace=True)
    return features[indices], labels[indices]

# ----------------- Step 11: 特征子集随机选取 -----------------
def feature_subset(num_features, num_to_pick, rng):
    """无放回随机抽取 num_to_pick 个不重复特征列索引。"""
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ----------------- Step 12: 随机森林训练 -----------------
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """训练随机森林：对每棵树执行行 Bootstrap 与列特征抽样并拟合。"""
    rng = np.random.default_rng(random_state)
    d = features.shape[1]
    k = max(1, int(round(np.sqrt(d)))) if num_features_per_split is None else num_features_per_split

    forest = []
    for _ in range(num_trees):
        boot_X, boot_y = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, k, rng)
        tree = build_tree(boot_X, boot_y, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feat_indices)
        forest.append({'tree': tree, 'feature_indices': feat_indices})
    return forest

# ----------------- Step 13: 多数投票聚合预测 -----------------
def combine_predictions(tree_predictions):
    """对森林中所有树的预测结果按样本列进行多数投票（Majority Vote）。"""
    tp = np.asarray(tree_predictions)
    n = tp.shape[1]
    out = np.zeros(n, dtype=int)
    for j in range(n):
        col = tp[:, j]
        vals, counts = np.unique(col, return_counts=True)
        out[j] = vals[np.argmax(counts)]
    return out

# ----------------- Step 14: 随机森林批量预测 -----------------
def predict_forest(forest, features):
    """使用训练完成的随机森林对特征数据集进行联合预测。"""
    tp = np.array([predict_tree(t['tree'], features) for t in forest])
    return combine_predictions(tp)

# ----------------- Step 15: 准确率评估 -----------------
def accuracy(predictions, labels):
    """计算预测标签与真实标签的匹配比率。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float(np.mean(predictions == labels))

# Step 5 - should_stop
import numpy as np

# ----------------- Step 1: 不纯度计算 (Gini Impurity) -----------------
def impurity(labels):
    """计算一维整型类别标签的基尼不纯度（0 为纯净，越混合值越大）。"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))

# ----------------- Step 2: 划分数据集 -----------------
def split_dataset(features, labels, feature_index, threshold):
    """根据指定特征与阈值将样本切分为左右两组 (左 <= threshold, 右 > threshold)。"""
    mask = features[:, feature_index] <= threshold
    return features[mask], labels[mask], features[~mask], labels[~mask]

# ----------------- Step 3: 分裂质量评分 (加权不纯度降低量) -----------------
def split_score(parent_labels, left_labels, right_labels):
    """计算分裂带来的不纯度减少量（信息增益），值越大表示分裂后越纯。"""
    n = len(parent_labels)
    if n == 0 or len(left_labels) == 0 or len(right_labels) == 0:
        return 0.0
    p_imp = impurity(parent_labels)
    l_imp = impurity(left_labels)
    r_imp = impurity(right_labels)
    weighted_child_imp = (len(left_labels) / n) * l_imp + (len(right_labels) / n) * r_imp
    return float(p_imp - weighted_child_imp)

# ----------------- Step 4: 寻找最优划分 -----------------
def best_split(features, labels, feature_indices):
    """在给定的候选特征子集中寻找能够最大化纯度提升的 (特征索引, 阈值)。"""
    best_score = 0.0
    best_feat = None
    best_thresh = None

    for f_idx in feature_indices:
        col_vals = features[:, f_idx]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 1:
            continue
        # 取相邻有序值的平均值作为候选切分点
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for th in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, th)
            score = split_score(labels, left_y, right_y)
            if score > best_score:
                best_score = score
                best_feat = f_idx
                best_thresh = th

    if best_score <= 1e-9 or best_feat is None:
        return {'feature_index': None, 'threshold': None, 'score': 0.0}
    return {
        'feature_index': int(best_feat),
        'threshold': float(best_thresh),
        'score': float(best_score)
    }

# ----------------- Step 5: 停止分裂判定 -----------------
def should_stop(labels, depth, max_depth, min_samples_split):
    """判定当前节点是否应当停止生长转为叶子节点。"""
    if len(labels) == 0 or len(np.unique(labels)) <= 1:
        return True
    if depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# ----------------- Step 6: 叶子节点类别预测 -----------------
def leaf_prediction(labels):
    """返回叶子节点样本集中出现频次最高的多数类别标签。"""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])

# ----------------- Step 7: 递归构建单棵决策树 -----------------
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    """递归构建决策树，返回嵌套字典结构的节点。"""
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    d = features.shape[1]
    cand_indices = list(range(d)) if feature_subset is None else feature_subset
    split = best_split(features, labels, cand_indices)
    
    if split['score'] <= 0.0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    feat_idx = split['feature_index']
    thresh = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feat_idx, thresh)
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}

    left_child = build_tree(left_X, left_y, max_depth=max_depth, min_samples_split=min_samples_split,
                            feature_subset=feature_subset, depth=depth + 1)
    right_child = build_tree(right_X, right_y, max_depth=max_depth, min_samples_split=min_samples_split,
                             feature_subset=feature_subset, depth=depth + 1)

    return {
        'leaf': False,
        'feature_index': int(feat_idx),
        'threshold': float(thresh),
        'left': left_child,
        'right': right_child
    }

# ----------------- Step 8: 单个样本决策树推理 -----------------
def predict_example_tree(tree, example):
    """沿决策树向下路由单个样本直到叶子节点，返回预测值。"""
    curr = tree
    while not curr['leaf']:
        if example[curr['feature_index']] <= curr['threshold']:
            curr = curr['left']
        else:
            curr = curr['right']
    return int(curr['prediction'])

# ----------------- Step 9: 批量数据决策树推理 -----------------
def predict_tree(tree, features):
    """使用单棵决策树对整个特征矩阵的所有样本进行批量预测。"""
    return np.array([predict_example_tree(tree, row) for row in features], dtype=int)

# ----------------- Step 10: 自助采样 (Bootstrap Sampling) -----------------
def bootstrap_sample(features, labels, rng):
    """有放回随机抽取与原样本量相同数量的行样本。"""
    n = len(features)
    indices = rng.choice(n, size=n, replace=True)
    return features[indices], labels[indices]

# ----------------- Step 11: 特征子集随机选取 -----------------
def feature_subset(num_features, num_to_pick, rng):
    """无放回随机抽取 num_to_pick 个不重复特征列索引。"""
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ----------------- Step 12: 随机森林训练 -----------------
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    """训练随机森林：对每棵树执行行 Bootstrap 与列特征抽样并拟合。"""
    rng = np.random.default_rng(random_state)
    d = features.shape[1]
    k = max(1, int(round(np.sqrt(d)))) if num_features_per_split is None else num_features_per_split

    forest = []
    for _ in range(num_trees):
        boot_X, boot_y = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, k, rng)
        tree = build_tree(boot_X, boot_y, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feat_indices)
        forest.append({'tree': tree, 'feature_indices': feat_indices})
    return forest

# ----------------- Step 13: 多数投票聚合预测 -----------------
def combine_predictions(tree_predictions):
    """对森林中所有树的预测结果按样本列进行多数投票（Majority Vote）。"""
    tp = np.asarray(tree_predictions)
    n = tp.shape[1]
    out = np.zeros(n, dtype=int)
    for j in range(n):
        col = tp[:, j]
        vals, counts = np.unique(col, return_counts=True)
        out[j] = vals[np.argmax(counts)]
    return out

# ----------------- Step 14: 随机森林批量预测 -----------------
def predict_forest(forest, features):
    """使用训练完成的随机森林对特征数据集进行联合预测。"""
    tp = np.array([predict_tree(t['tree'], features) for t in forest])
    return combine_predictions(tp)

# ----------------- Step 15: 准确率评估 -----------------
def accuracy(predictions, labels):
    """计算预测标签与真实标签的匹配比率。"""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return float(np.mean(predictions == labels))

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

