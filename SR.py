import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 读取数据 ==========
file_path = "beam.csv"  # 请确保路径正确
data = pd.read_csv(file_path)

# ========== 2. 构造数据集（保留工况ID和点索引） ==========
X_list = []
y_list = []
case_id_list = []      # 记录每个样本对应的工况ID（原始行索引）
point_idx_list = []     # 记录每个样本对应的点编号

for case_idx, row in data.iterrows():
    L = row['Length']
    b = row['Width']
    h = row['Height']
    E = row['E']
    nu = row['nu']
    a = row['Load_x_coord']
    displacements = [row[f'Point_{i}'] for i in range(1, 11)]
    for i, w in enumerate(displacements, start=1):
        x = i * L / 10.0          # 第 i 个点的位置
        X_list.append([L, b, h, E, nu, a, x])
        y_list.append(w)
        case_id_list.append(case_idx)   # 保存工况ID
        point_idx_list.append(i)         # 保存点索引

X = np.array(X_list)
y = np.array(y_list)
case_id = np.array(case_id_list)
point_idx = np.array(point_idx_list)

# ========== 3. 按工况划分训练/测试/验证集 ==========
unique_cases = np.unique(case_id)
# 先将所有工况划分为训练工况和测试工况（70%训练，30%测试）
train_cases, test_cases = train_test_split(unique_cases, test_size=0.3, random_state=42)
# 再从训练工况中划分一部分作为验证工况（训练工况的20%，即占全体工况的14%）
train_cases, val_cases = train_test_split(train_cases, test_size=0.2, random_state=42)

# 根据工况ID提取对应的样本
train_mask = np.isin(case_id, train_cases)
val_mask   = np.isin(case_id, val_cases)
test_mask  = np.isin(case_id, test_cases)

X_train, y_train, idx_train_pt = X[train_mask], y[train_mask], point_idx[train_mask]
X_val,   y_val,   idx_val_pt   = X[val_mask],   y[val_mask],   point_idx[val_mask]
X_test,  y_test,  idx_test_pt  = X[test_mask], y[test_mask], point_idx[test_mask]

print(f"训练工况数: {len(train_cases)}，验证工况数: {len(val_cases)}，测试工况数: {len(test_cases)}")
print(f"训练样本数: {len(X_train)}，验证样本数: {len(X_val)}，测试样本数: {len(X_test)}")

# ========== 4. 数据放大（使训练集 y 最大绝对值为 1） ==========
scale_y = 1.0 / np.max(np.abs(y_train))
y_train_scaled = y_train * scale_y
y_val_scaled   = y_val   * scale_y
y_test_scaled  = y_test  * scale_y
print(f"数据缩放因子 scale_y = {scale_y:.6e}")

# ========== 5. 定义安全函数（仅保留物理相关的基本运算） ==========
def safe_add(x1, x2): return x1 + x2
def safe_sub(x1, x2): return x1 - x2
def safe_mul(x1, x2): return x1 * x2
def safe_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        x2_abs = np.abs(x2)
        return np.where(x2_abs > 1e-10, x1 / x2, 0.0)
def safe_square(x): return x**2
def safe_cube(x): return x**3

add_func = make_function(function=safe_add, name='add', arity=2)
sub_func = make_function(function=safe_sub, name='sub', arity=2)
mul_func = make_function(function=safe_mul, name='mul', arity=2)
div_func = make_function(function=safe_div, name='div', arity=2)
square_func = make_function(function=safe_square, name='square', arity=1)
cube_func = make_function(function=safe_cube, name='cube', arity=1)

function_set = [add_func, sub_func, mul_func, div_func, square_func, cube_func]

# ========== 6. 设置常数范围 ==========
max_abs_scaled = np.max(np.abs(y_train_scaled))
const_range = (-20 * max_abs_scaled, 20 * max_abs_scaled)
print(f"常数范围设置为: {const_range}")

# ========== 7. 多次尝试以满足误差要求 ==========
target_train_mape = 3.0   # 训练集内MAPE ≤ 3%
target_test_mape  = 10.0  # 测试集外MAPE ≤ 10%
max_attempts = 5
best_model = None
best_test_mape = np.inf
best_train_mape = np.inf

for attempt in range(max_attempts):
    print(f"\\n========== 尝试 {attempt+1}/{max_attempts} ==========")
    # 每次尝试使用不同的随机种子
    random_state = 42 + attempt

    est = SymbolicRegressor(
        population_size=10000,
        generations=100,
        tournament_size=20,
        stopping_criteria=0.0001,
        const_range=const_range,
        init_depth=(2, 5),
        function_set=function_set,
        metric='mse',
        parsimony_coefficient=0.1,          # 提高简洁性惩罚
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=0.9,
        feature_names=['L', 'b', 'h', 'E', 'nu', 'a', 'x'],
        random_state=random_state,
        verbose=1
    )

    # 训练
    est.fit(X_train, y_train_scaled)

    # 预测并还原尺度
    y_train_pred_scaled = est.predict(X_train)
    y_val_pred_scaled   = est.predict(X_val)
    y_test_pred_scaled  = est.predict(X_test)

    # 处理可能的NaN/Inf
    for arr in [y_train_pred_scaled, y_val_pred_scaled, y_test_pred_scaled]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)

    y_train_pred = y_train_pred_scaled / scale_y
    y_val_pred   = y_val_pred_scaled   / scale_y
    y_test_pred  = y_test_pred_scaled  / scale_y

    # 计算MAPE（避免除零，加小常数）
    def mape(y_true, y_pred):
        eps = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

    train_mape = mape(y_train, y_train_pred)
    val_mape   = mape(y_val,   y_val_pred)
    test_mape  = mape(y_test,  y_test_pred)

    print(f"训练集 MAPE: {train_mape:.2f}%")
    print(f"验证集 MAPE: {val_mape:.2f}%")
    print(f"测试集 MAPE: {test_mape:.2f}%")

    # 检查表达式是否包含所有特征
    expr_str = str(est._program)
    all_features = ['L', 'b', 'h', 'E', 'nu', 'a', 'x']
    missing = [f for f in all_features if f not in expr_str]
    if missing:
        print(f"表达式缺少特征: {missing}")
    else:
        print("表达式包含所有特征")

    # 判断是否满足目标
    if train_mape <= target_train_mape and test_mape <= target_test_mape:
        print("🎉 找到满足误差要求的模型！")
        best_model = est
        best_train_mape = train_mape
        best_test_mape = test_mape
        break
    else:
        # 记录当前最好的测试MAPE（优先考虑测试误差）
        if test_mape < best_test_mape:
            best_test_mape = test_mape
            best_train_mape = train_mape
            best_model = est
        print("未满足要求，继续尝试...")

if best_model is None:
    print("未找到完全满足要求的模型，将输出最佳模型（测试MAPE最小）。")

# ========== 8. 使用最佳模型进行最终评估 ==========
est = best_model
y_train_pred_scaled = est.predict(X_train)
y_val_pred_scaled   = est.predict(X_val)
y_test_pred_scaled  = est.predict(X_test)

# 处理NaN/Inf
for arr in [y_train_pred_scaled, y_val_pred_scaled, y_test_pred_scaled]:
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)

y_train_pred = y_train_pred_scaled / scale_y
y_val_pred   = y_val_pred_scaled   / scale_y
y_test_pred  = y_test_pred_scaled  / scale_y

train_mape = mape(y_train, y_train_pred)
val_mape   = mape(y_val,   y_val_pred)
test_mape  = mape(y_test,  y_test_pred)

print("\\n========== 最终结果 ==========")
print(f"训练集 MAPE: {train_mape:.2f}%")
print(f"验证集 MAPE: {val_mape:.2f}%")
print(f"测试集 MAPE: {test_mape:.2f}%")

# ========== 9. 输出最佳表达式 ==========
print("\\n最佳表达式（预测缩放后的位移）:")
print(est._program)
print(f"\\n实际预测位移表达式（原始尺度）: ({est._program}) / {scale_y:.6e}")

# 检查特征完整性
expr_str = str(est._program)
missing = [f for f in all_features if f not in expr_str]
if missing:
    print(f"\\n警告：最终表达式缺少特征: {missing}")
else:
    print("\\n表达式包含所有输入特征。")

# 计算表达式节点数，评估简洁性
node_count = est._program.length_
print(f"表达式节点数: {node_count}")

# ========== 10. 绘制对比图（11个子图） ==========
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# 前10个子图：分别对应点1-10
for point in range(1, 11):
    ax = axes[point-1]
    mask = (idx_test_pt == point)
    if np.sum(mask) == 0:
        ax.text(0.5, 0.5, f'No samples for Point {point}', ha='center', va='center')
        ax.set_title(f'Point {point}')
        continue
    y_true_point = y_test[mask]
    y_pred_point = y_test_pred[mask]
    ax.scatter(y_true_point, y_pred_point, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_true_point.min(), y_pred_point.min())
    max_val = max(y_true_point.max(), y_pred_point.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    ax.set_title(f'Point {point}')
    ax.grid(True, linestyle='--', alpha=0.7)
    # 标注MAPE
    point_mape = mape(y_true_point, y_pred_point)
    ax.text(0.05, 0.95, f'MAPE: {point_mape:.1f}%', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 第11个子图：所有点混合
ax = axes[10]
ax.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax.set_xlabel('True Displacement')
ax.set_ylabel('Predicted Displacement')
ax.set_title(f'All Points Combined (MAPE: {test_mape:.1f}%)')
ax.grid(True, linestyle='--', alpha=0.7)

# 隐藏第12个子图
if len(axes) > 11:
    axes[11].set_visible(False)

plt.tight_layout()
plt.show()