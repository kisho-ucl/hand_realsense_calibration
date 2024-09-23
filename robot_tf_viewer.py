import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

n=36

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
    ax.quiver(*origin, *x_axis, color='r', length=0.5, normalize=True)
    ax.quiver(*origin, *y_axis, color='g', length=0.5, normalize=True)
    ax.quiver(*origin, *z_axis, color='b', length=0.5, normalize=True)

    # 新しい座標系のプロット
    ax.quiver(*new_origin, *new_x, color='r', length=0.0, normalize=True)
    ax.quiver(*new_origin, *new_y, color='g', length=0.0, normalize=True)
    ax.quiver(*new_origin, *new_z, color='b', length=0.2, normalize=True)

    # 軸ラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text(*new_origin, label, color='black')

    # 軸の範囲を揃える
    scale = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    max_range = (scale[:, 1] - scale[:, 0]).max() / 2.0

    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())

    #ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

def params2trans(params):
    quaternion = [0.5,-0.5,0.5,0.5]
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    translation_vector = np.array([0.0, params[0], params[1]])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # 回転行列を代入
    transformation_matrix[:3, 3] = translation_vector  # 平行移動ベクトルを代入
    return transformation_matrix

def trans2param(matrix, seq='xyz'):
    # 位置ベクトルを抽出
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    # 回転行列部分を抽出
    rotation_matrix = matrix[:3, :3]
    # 回転行列をRotationオブジェクトに変換し、オイラー角を計算
    rot = R.from_matrix(rotation_matrix)
    euler_angles = rot.as_euler(seq, degrees=False)
    # 位置とオイラー角を返す
    position = (x, y, z)
    return np.array(position), np.array(euler_angles)

def pos_angle_str(pos,angle):
    pos_mm = (pos * 1000).astype(int)
    angle_deg = (angle*180/np.pi).astype(int)
    robot_str = []
    for p in pos_mm: 
        robot_str.append(p)
    for a in angle_deg:
        robot_str.append(a)
    return robot_str


T1_list = []
T2_list = []
T3_list = []
T4_list = []
for i in range(8):
    dir0 = f'/Users/kisho/open3d/Experiments/dataset_paramEst_auto4/data{i}'
    T1 = np.load(f"{dir0}/trans1.npy")
    T2 = np.load(f"{dir0}/trans2.npy")
    T3 = np.load(f"{dir0}/trans3.npy")
    T4 = np.load(f"{dir0}/trans4.npy")
    T1_list.append(T1)
    T2_list.append(T2)
    T3_list.append(T3)
    T4_list.append(T4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(8):
    T = T1_list[i]
    pos,angle = trans2param(T)
    robot_str = pos_angle_str(pos,angle)
    print(robot_str)
    plot_transform(T, ax, label=f'{i}')

plt.show()
