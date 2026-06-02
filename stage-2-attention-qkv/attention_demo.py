"""
阶段 2 动手任务：手动实现 Self-Attention 的完整计算流程
句子："我 吃 苹果"，向量维度 d_k = 4

目标：计算"吃"这个词经过注意力机制后的新向量

不依赖任何第三方库，纯 Python 实现
"""

import math

# ============================================================
# 第一步：定义输入矩阵
# ============================================================
# Q、K、V 是模型通过 W_Q、W_K、W_V 对输入 X 变换得到的
# 这里我们直接给定结果，跳过变换过程（阶段 2 重点是注意力计算本身）

# Q 矩阵：3 个词 × 4 维（每一行是一个词的 Query 向量）
# Q = "我想问什么"
Q = [
    [1.0, 0.0, 1.0, 0.0],   # "我" 的 Query
    [0.0, 1.0, 0.0, 1.0],   # "吃" 的 Query
    [1.0, 1.0, 0.0, 0.0],   # "苹果" 的 Query
]

# K 矩阵：3 个词 × 4 维（每一行是一个词的 Key 向量）
# K = "我能提供什么线索让别人找到我"
K = [
    [1.0, 0.0, 0.0, 1.0],   # "我" 的 Key
    [0.0, 1.0, 1.0, 0.0],   # "吃" 的 Key
    [1.0, 0.0, 1.0, 0.0],   # "苹果" 的 Key
]

# V 矩阵：3 个词 × 4 维（每一行是一个词的 Value 向量）
# V = "我的实际内容，被别人引用时用什么"
V = [
    [0.1, 0.2, 0.3, 0.4],   # "我" 的 Value
    [0.5, 0.6, 0.7, 0.8],   # "吃" 的 Value
    [0.9, 1.0, 1.1, 1.2],   # "苹果" 的 Value
]

words = ["我", "吃", "苹果"]

def print_matrix(name, M):
    """格式化打印矩阵"""
    print(f"{name} =")
    for i, row in enumerate(M):
        print(f"  [{', '.join(f'{v:.1f}' for v in row)}]  ← '{words[i]}'")

print("=" * 60)
print("输入矩阵（3 个词，4 维向量）")
print("=" * 60)
print_matrix("Q (Query = '我想问什么')", Q)
print()
print_matrix("K (Key   = '我的线索')", K)
print()
print_matrix("V (Value = '我的内容')", V)

# ============================================================
# 第二步：计算 Q 和 K 的点积（相似度分数）
# ============================================================
# Q @ K.T 的含义：
#   - Q 的每一行（每个词的 Query）和 K 的每一行（每个词的 Key）做点积
#   - 结果是一个 3×3 的矩阵：scores[i][j] = 词 i 的 Query 和 词 j 的 Key 的匹配度
#
# 点积的几何意义：
#   a · b = |a| × |b| × cos(θ)
#   → 方向一致（θ≈0）→ cos(θ)≈1 → 点积大 → "这两个词相关"
#   → 方向垂直（θ=90°）→ cos(θ)=0 → 点积为零 → "无关"
#   → 方向相反（θ=180°）→ cos(θ)≈-1 → 点积最负 → "冲突"

def dot(a, b):
    """两个向量的点积"""
    return sum(x * y for x, y in zip(a, b))

def mat_mul_AT_B(A, B):
    """计算 A @ B^T，结果[i][j] = A的第i行 · B的第j行"""
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            row.append(dot(A[i], B[j]))
        result.append(row)
    return result

scores = mat_mul_AT_B(Q, K)

print(f"\n{'=' * 60}")
print("第一步：点积计算  Q @ K^T")
print(f"{'=' * 60}")
print_matrix("scores", scores)
print(f"\n解读这个 3×3 矩阵：")
for i in range(3):
    for j in range(3):
        print(f"  scores[{i}][{j}] = {scores[i][j]:.1f}  → '{words[i]}'的Q 和 '{words[j]}'的K 的匹配度")

# 手动验证一个点积：
# "吃"的Q = [0, 1, 0, 1]，"苹果"的K = [1, 0, 1, 0]
# 点积 = 0×1 + 1×0 + 0×1 + 1×0 = 0
manual_dot = dot(Q[1], K[2])
print(f"\n手动验证：'吃'的Q · '苹果'的K = {manual_dot}")
print(f"  → 0 说明方向垂直，完全无关（在这个简化例子里）")

# ============================================================
# 第三步：除以 √d_k（缩放）
# ============================================================
# 为什么需要缩放？
#   - 维度越高，点积的绝对值天然越大（更多维度累加）
#   - 大数值扔进 Softmax → 输出接近 one-hot → 梯度消失 → 学不动
#   - 除以 √d_k 把数值拉回合理范围
#
# 这里 d_k = 4，所以 √d_k = 2

d_k = len(K[0])              # K 的列数 = 向量维度 = 4
scale = math.sqrt(d_k)       # √4 = 2.0

def scale_matrix(M, factor):
    """矩阵每个元素除以 factor"""
    return [[v / factor for v in row] for row in M]

scaled_scores = scale_matrix(scores, scale)

print(f"\n{'=' * 60}")
print(f"第二步：缩放  scores / √d_k = scores / √{d_k} = scores / {scale}")
print(f"{'=' * 60}")
print_matrix("scaled_scores", scaled_scores)

# 对比缩放前后
max_before = max(max(row) for row in scores)
max_after = max(max(row) for row in scaled_scores)
print(f"\n对比缩放前后：")
print(f"  缩放前最大值: {max_before:.1f}")
print(f"  缩放后最大值: {max_after:.1f}")
print(f"  → 数值变小了，Softmax 不会被撑爆")

# ============================================================
# 第四步：Softmax 归一化（变成概率分布）
# ============================================================
# Softmax 的作用：把任意实数 → 概率分布（每行加起来 = 1）
# 公式：softmax(x_i) = e^(x_i) / Σ e^(x_j)
#
# 注意：这里是对每一行单独做 Softmax
# 因为每一行代表"一个词对所有词的关注度分配"

def softmax_row(row):
    """对一行做 Softmax，带数值稳定处理"""
    # 减去最大值防止 exp 溢出（这是工程技巧，不影响结果）
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def softmax(M):
    """对矩阵每一行做 Softmax"""
    return [softmax_row(row) for row in M]

attention_weights = softmax(scaled_scores)

print(f"\n{'=' * 60}")
print(f"第三步：Softmax 归一化")
print(f"{'=' * 60}")
print_matrix("attention_weights", attention_weights)

print(f"\n验证每行加起来是否为 1：")
for i, word in enumerate(words):
    row_sum = sum(attention_weights[i])
    print(f"  '{word}' 那行: [{', '.join(f'{v:.4f}' for v in attention_weights[i])}]  → 总和 = {row_sum:.4f}")

print(f"\n解读权重（以'吃'为例）：")
print(f"  '吃'对'我'的关注度:   {attention_weights[1][0]:.4f}")
print(f"  '吃'对'吃'自身的关注度: {attention_weights[1][1]:.4f}")
print(f"  '吃'对'苹果'的关注度: {attention_weights[1][2]:.4f}")

# ============================================================
# 第五步：用权重对 V 做加权求和
# ============================================================
# 最终输出 = attention_weights @ V
# 含义：每个词的新向量 = 所有词的 Value 按关注度权重的混合
#
# 例：如果"吃"对"苹果"的权重是 0.5，对"我"的权重是 0.3，对自身的权重是 0.2
# 那么"吃"的新向量 = 0.3×V_我 + 0.2×V_吃 + 0.5×V_苹果

def mat_mul(A, B):
    """矩阵乘法 A @ B，A 的行数 × B 的列数"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "矩阵维度不匹配！"
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

output = mat_mul(attention_weights, V)

print(f"\n{'=' * 60}")
print(f"第四步：加权求和  attention_weights @ V")
print(f"{'=' * 60}")
print(f"输出矩阵 =")
for i, word in enumerate(words):
    print(f"  [{', '.join(f'{v:.4f}' for v in output[i])}]  ← '{word}' 的新向量")

# ============================================================
# 汇总：只看"吃"的完整计算过程
# ============================================================
print(f"\n{'=' * 60}")
print(f"汇总：'吃' 的注意力计算全过程")
print(f"{'=' * 60}")
print(f"1. '吃'的 Query:         {Q[1]}")
print(f"2. 与所有 Key 的点积:     {scores[1]}")
print(f"3. 缩放后:               {scaled_scores[1]}")
print(f"4. Softmax 权重:          [{', '.join(f'{v:.4f}' for v in attention_weights[1])}]")
print(f"5. 加权混合所有 Value:    [{', '.join(f'{v:.4f}' for v in output[1])}]")
print(f"\n→ '吃'的新向量是三个词 Value 的加权混合")
print(f"   权重由'吃'的 Query 和每个词的 Key 的匹配度决定")

# ============================================================
# 完整流程一句话总结
# ============================================================
print(f"\n{'=' * 60}")
print("完整流程一句话总结")
print(f"{'=' * 60}")
print("""
Q = "我想找什么"    →  每个词发出查询
K = "我能提供什么"  →  每个词亮出标签
V = "我的实际内容"  →  每个词的真正信息

Q·K^T        →  算谁和谁匹配
/ √d_k       →  降温，防止 Softmax 爆炸
Softmax      →  变成概率权重
× V          →  按权重混合内容

输出 = 每个词融合了上下文信息后的新向量
""")
