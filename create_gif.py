"""
create_gif.py — 从 OBJ 序列生成球体→奶牛变形动画 GIF

用法:
    conda run -n torch python create_gif.py

依赖: numpy, matplotlib, Pillow, imageio
    conda install -n torch imageio  (如未安装)
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

# =========================================================
# 配置
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# OBJ 文件按 epoch 排序
obj_files = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "mesh_epoch_*.obj")),
    key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0])
)

print(f"找到 {len(obj_files)} 个 OBJ 文件")

# GIF 输出路径
GIF_PATH = os.path.join(OUTPUT_DIR, "animation.gif")

# 渲染参数
IMG_SIZE = (640, 480)        # 输出图片尺寸
ELEV, AZIM = 25, -60        # 固定相机视角 (仰角, 方位角)
DIST = 3.0                   # 相机距离

# =========================================================
# OBJ 解析器
# =========================================================

def load_obj_verts_faces(obj_path):
    """从 OBJ 文件读取顶点和面片"""
    verts = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # 处理 f v1 v2 v3 或 f v1/t1 v2/t2 v3/t3
                face_verts = []
                for p in parts[1:]:
                    idx = p.split("/")[0]
                    face_verts.append(int(idx) - 1)  # OBJ 是 1-indexed
                # 只取三角形
                if len(face_verts) >= 3:
                    faces.append(face_verts[:3])
    return np.array(verts), np.array(faces)


def compute_face_normals(verts, faces):
    """计算每个面的法线"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def compute_vertex_displacement(all_verts):
    """计算每个 epoch 相对于初始状态的顶点位移量（用于着色）"""
    base = all_verts[0]
    displacements = []
    for v in all_verts:
        disp = np.mean(np.linalg.norm(v - base, axis=1))
        displacements.append(disp)
    return np.array(displacements)


# =========================================================
# 渲染函数
# =========================================================

def render_mesh(verts, faces, ax, elev=25, azim=-60, dist=3.0, face_colors=None):
    """在指定 Axes 上渲染网格"""
    ax.clear()

    # 设置视角
    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist

    # 计算面法线用于光照
    normals = compute_face_normals(verts, faces)

    # 光源方向（从上左前方）
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Lambertian 漫反射光照
    cos_angles = np.clip(np.dot(normals, light_dir), 0.1, 1.0)

    # 基础颜色 + 光照
    if face_colors is None:
        base_color = np.array([0.4, 0.7, 1.0])  # 浅蓝色
    else:
        base_color = face_colors

    colors = base_color * cos_angles[:, np.newaxis]
    colors = np.clip(colors, 0, 1)

    # 创建面片集合
    mesh = Poly3DCollection(
        [verts[f] for f in faces],
        facecolors=colors,
        edgecolors=(0.2, 0.2, 0.2, 0.3),
        linewidths=0.1,
        antialiased=True
    )

    ax.add_collection3d(mesh)

    # 设置坐标轴范围
    margin = 0.2
    x_min, x_max = verts[:, 0].min() - margin, verts[:, 0].max() + margin
    y_min, y_max = verts[:, 1].min() - margin, verts[:, 1].max() + margin
    z_min, z_max = verts[:, 2].min() - margin, verts[:, 2].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 统一坐标轴比例
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 隐藏坐标轴
    ax.set_axis_off()

    return mesh


# =========================================================
# 主流程
# =========================================================

def main():
    # 1. 加载所有 OBJ
    print("正在加载 OBJ 文件...")
    all_verts = []
    faces = None
    for f in obj_files:
        v, f_local = load_obj_verts_faces(f)
        all_verts.append(v)
        if faces is None:
            faces = f_local
        assert faces.shape == f_local.shape, f"面片拓扑不一致: {f}"

    verts_array = np.array(all_verts)  # (N_frames, N_verts, 3)
    N_frames = len(all_verts)
    print(f"加载完成: {N_frames} 帧, {verts_array.shape[1]} 顶点, {faces.shape[0]} 面片")

    # 2. 线性插值生成中间帧（使动画更平滑）
    interp_factor = 3  # 每两个关键帧之间插入 3 帧
    print(f"正在插值 (factor={interp_factor})...")
    interpolated_verts = []
    for i in range(N_frames - 1):
        for t in np.linspace(0, 1, interp_factor + 1)[:-1]:
            v = (1 - t) * verts_array[i] + t * verts_array[i + 1]
            interpolated_verts.append(v)
    # 添加最后一帧
    interpolated_verts.append(verts_array[-1])
    verts_smooth = np.array(interpolated_verts)
    N_smooth = len(verts_smooth)
    print(f"插值后: {N_smooth} 帧")

    # 3. 渲染帧
    print("正在渲染帧...")
    os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)

    fig = plt.figure(figsize=(8, 6), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    frame_paths = []

    for idx in range(N_smooth):
        if idx % 20 == 0:
            print(f"  渲染帧 {idx + 1}/{N_smooth}")

        verts = verts_smooth[idx]

        # 根据变形程度映射颜色（从蓝色渐变到橙色）
        progress = idx / (N_smooth - 1) if N_smooth > 1 else 0
        r = 0.2 + 0.6 * progress
        g = 0.5 + 0.3 * (1 - progress)
        b = 1.0 - 0.6 * progress
        base_color = np.array([r, g, b])

        render_mesh(verts, faces, ax, elev=ELEV, azim=AZIM, dist=DIST,
                    face_colors=base_color)

        # 添加标题
        ax.set_title(
            f"Epoch {idx * 50 // (interp_factor + 1) if idx < N_smooth - 1 else 799}",
            color="white", fontsize=14, pad=10
        )

        # 保存帧
        frame_path = os.path.join(OUTPUT_DIR, "frames", f"frame_{idx:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches="tight", pad_inches=0.1,
                    facecolor="#1a1a2e")
        frame_paths.append(frame_path)

    plt.close(fig)
    print("渲染完成!")

    # 4. 合成 GIF
    print("正在合成 GIF...")
    images = []
    for path in frame_paths:
        img = Image.open(path)
        images.append(img)

    # 保存 GIF
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=60,  # 每帧显示 60ms (~16.7 FPS)
        loop=0,       # 无限循环
        optimize=True
    )
    print(f"GIF 已保存: {GIF_PATH}")

    # 5. 清理临时帧文件（可选）
    import shutil
    shutil.rmtree(os.path.join(OUTPUT_DIR, "frames"))
    print("临时帧文件已清理")


if __name__ == "__main__":
    main()
