import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

n=36
dir0 = '/Users/kisho/open3d/hand-realsense-calibration/data_36'

def plot_transform(T, ax=None, label=''):
    # 回転部分と平行移動部分を抽出
    R = T[:3, :3]  # 3x3の回転行列
    t = T[:3, 3]   # 3x1の平行移動ベクトル

    # 基準座標系の軸
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # 回転行列に基づき、新しい座標軸を計算
    new_x = R @ x_axis
    new_y = R @ y_axis
    new_z = R @ z_axis

    # 軸を移動させるためのオフセットを追加
    new_origin = origin + t

    # プロット
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # 元の座標系のプロット
    ax.quiver(*origin, *x_axis, color='r', length=1, normalize=True)
    ax.quiver(*origin, *y_axis, color='g', length=1, normalize=True)
    ax.quiver(*origin, *z_axis, color='b', length=1, normalize=True)

    # 新しい座標系のプロット
    ax.quiver(*new_origin, *new_x, color='r', length=1, normalize=True)
    ax.quiver(*new_origin, *new_y, color='g', length=1, normalize=True)
    ax.quiver(*new_origin, *new_z, color='b', length=1, normalize=True)

    # 軸ラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text(*new_origin, label, color='black')

    # 軸の範囲を揃える
    scale = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    max_range = (scale[:, 1] - scale[:, 0]).max() #/ 2.0

    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())

    #ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)

def params2trans(params):
    quaternion = [0.5,-0.5,0.5,0.5]
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    translation_vector = np.array([0.0, params[0], params[1]])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # 回転行列を代入
    transformation_matrix[:3, 3] = translation_vector  # 平行移動ベクトルを代入
    return transformation_matrix



T1_list = []
T2_list = []
T3_list = []
T4_list = []
T_cam_list = []
T_brd_on_map_list = []
trans_list = []


for i in range(n):
    T1 = np.load(f"{dir0}/T1_{i}.npy")
    T2 = np.load(f"{dir0}/T2_{i}.npy")
    T3 = np.load(f"{dir0}/T3_{i}.npy")
    T4 = np.load(f"{dir0}/T4_{i}.npy")
    T2 = params2trans([0, 0.036])
    T_cam = T1@T2@T3
    T_brd_on_map = T_cam@T4
    T1_list.append(T1)
    T2_list.append(T2)
    T3_list.append(T3)
    T4_list.append(T4)
    T_cam_list.append(T_cam)
    T_brd_on_map_list.append(T_brd_on_map)
    #trans_list.append(T_brd_on_map)

def evaluate_function(params):
    trans_list = []
    for i in range(n):
        T1=T1_list[i]
        T2=params2trans(params)
        #print(T2)
        #T2=T2_list[i]
        #T2[0,3] = params[0]
        #T2[1,3] = params[1]
        #T2[2,3] = params[2]
        T3=T3_list[i]
        T4=T4_list[i]
        trans = T1@T2@T3@T4
        trans_list.append(trans)

     

    translation_vectors = np.array([trans[:3, 3] for trans in trans_list])  # 各行列の平行移動ベクトル部分
    rotation_matrices = np.array([trans[:3, :3] for trans in trans_list])   # 各行列の回転行列部分
    mean_translation = np.mean(translation_vectors, axis=0)
    quaternions = np.array([R.from_matrix(rot).as_quat() for rot in rotation_matrices])
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)  # 正規化
    translation_var = np.sum(np.linalg.norm(translation_vectors - mean_translation, axis=1) ** 2)/len(translation_vectors)
    translation_score = np.sqrt(translation_var)
    quaternion_score = 0
    for quat in quaternions:
        diff_quat = R.from_quat(mean_quaternion).inv() * R.from_quat(quat)
        angle = diff_quat.magnitude()  # クォータニオン間の回転角度
        quaternion_score += angle ** 2
 
    score = translation_score #+ quaternion_score
    
    return score



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rot = R.from_euler('xyz', [0.0, -np.pi/2, np.pi/2])
q0 = rot.as_quat()
# 初期値 (仮の値)
initial_params = np.array([0,0])

# 最適化
result = minimize(evaluate_function, initial_params)
optimized_params = result.x
print(evaluate_function(optimized_params))

print("最適化されたパラメータ:", optimized_params)
# 変換行列を可視化
#plot_transform(T1, ax, label='T1')
for i in range(n):
    #plot_transform(T_cam, ax, label='T_cam')
    T1=T1_list[i]
    T2=params2trans(result.x)
    #print(T2)
    T3=T3_list[i]
    T4=T4_list[i]
    T = T1@T2@T3@T4
    plot_transform(T, ax, label=f'Brd{i}')

print(T2)


plt.show()
