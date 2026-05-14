"""
Whitted-Style 光线追踪实验
========================================
支持环境：纯 CPU（无需 CUDA / Vulkan 显卡）
运行方式：python ray_tracing.py

实现内容：
  任务1 - 三种几何体（棋盘地面 + 漫反射红球 + 镜面银球）
  任务2 - 迭代式光线弹射（用 done 标志代替 break）
  任务3 - 硬阴影 + Shadow Acne 修复（法线方向偏移 EPS）
  任务4 - UI 交互面板（光源位置 + 最大弹射次数滑块）
"""

import taichi as ti

# ══════════════════════════════════════════════════════════════════════════════
#  初始化 Taichi
#  arch=ti.cpu  → 强制使用 CPU 后端，兼容所有设备
#  CPU 后端没有 Vulkan/CUDA 的"静态控制流"限制，
#  但我们依然保持 GPU 友好写法，方便未来迁移
# ══════════════════════════════════════════════════════════════════════════════
ti.init(arch=ti.cpu)

# ── 渲染分辨率 ────────────────────────────────────────────────────────────────
WIDTH  = 800   # 窗口宽度（像素）
HEIGHT = 600   # 窗口高度（像素）

# MAX_B：光线弹射循环的【静态上界】
# 必须是编译期常量，不能用运行时变量作为 range() 参数
# 用户在 UI 上调节的 max_bounces 只用来在循环内做 "bounce < max_b" 判断
MAX_B = 5

# ── Taichi 数据容器（field）────────────────────────────────────────────────────
# pixels      : 每个像素存一个 RGB 颜色向量
# light_pos   : 点光源的三维坐标，由 UI 滑块实时修改
# max_bounces : 用户设定的最大弹射次数（1~5）
pixels      = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos   = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())

# ── 场景几何常量 ───────────────────────────────────────────────────────────────
GROUND_Y        = -1.0                          # 地面高度（y = -1）
SPHERE_R        = 1.0                           # 两个球的半径
SPHERE_DIFF_POS = ti.math.vec3(-1.5, 0.0, 0.0) # 左边：红色漫反射球圆心
SPHERE_MIRR_POS = ti.math.vec3( 1.5, 0.0, 0.0) # 右边：银色镜面球圆心

EPS = 1e-4   # 防自相交偏移量（Shadow Acne 修复关键）
INF = 1e30   # 表示"无穷远"，用于初始化最近交点距离


# ══════════════════════════════════════════════════════════════════════════════
#  几何求交函数
# ══════════════════════════════════════════════════════════════════════════════

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """
    射线与球体求交（解析几何法）
    
    射线参数方程：P(t) = ro + t * rd
    球面方程：|P - center|² = radius²
    代入展开得一元二次方程：a·t² + b·t + c = 0
    
    参数：
        ro     : 射线起点 (ray origin)
        rd     : 射线方向 (ray direction，已归一化)
        center : 球心坐标
        radius : 球半径
    返回：
        hit    : 是否命中
        t      : 最近交点的参数值（距离）
        n      : 交点处的单位法向量（从球心指向交点）
    """
    oc   = ro - center           # 从球心到射线起点的向量
    a    = rd.dot(rd)            # = 1.0（rd 已归一化时）
    b    = 2.0 * oc.dot(rd)
    c    = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c  # 判别式：>0 两交点，=0 切线，<0 不相交

    hit = False
    t   = INF
    n   = ti.math.vec3(0.0, 1.0, 0.0)  # 默认法线（占位）

    if disc >= 0.0:
        sq = ti.sqrt(disc)
        t0 = (-b - sq) / (2.0 * a)   # 较小的根（射线先穿过这里）
        t1 = (-b + sq) / (2.0 * a)   # 较大的根（射线后穿过这里）

        # 优先取 t0（近端），若 t0 <= EPS（在起点背后或太近），再取 t1
        if t0 > EPS:
            t = t0
            hit = True
        elif t1 > EPS:
            t = t1
            hit = True

        if hit:
            p = ro + t * rd              # 计算交点坐标
            n = (p - center).normalized()  # 法线 = 从球心到交点的单位向量

    return hit, t, n


@ti.func
def intersect_ground(ro, rd):
    """
    射线与无限水平面（y = GROUND_Y）求交
    
    令 ro.y + t * rd.y = GROUND_Y，解出 t
    
    返回：
        hit : 是否命中（t > EPS 且射线朝平面方向）
        t   : 交点参数值
    """
    hit = False
    t   = INF

    # rd.y 接近 0 表示射线几乎平行于地面，不会相交
    if ti.abs(rd.y) > 1e-6:
        tc = (GROUND_Y - ro.y) / rd.y  # 解出 t
        if tc > EPS:                    # 交点在射线前方才有效
            hit = True
            t   = tc

    return hit, t


# ══════════════════════════════════════════════════════════════════════════════
#  场景整体求交
#  遍历所有物体，返回最近命中物体的信息
#
#  材质 ID 约定：
#    -1 = 未命中任何物体（射向天空）
#     0 = 地面（漫反射，棋盘格纹理）
#     1 = 左球（红色漫反射）
#     2 = 右球（银色镜面）
# ══════════════════════════════════════════════════════════════════════════════

@ti.func
def scene_intersect(ro, rd):
    """
    对场景中所有物体做求交，返回最近命中点的完整信息。
    
    返回：
        mat_id   : 材质 ID（见上方约定）
        hit_p    : 交点世界坐标
        hit_n    : 交点法向量（朝外）
        base_col : 物体基础颜色（或纹理颜色）
    """
    best_t   = INF                      # 当前最近距离，初始为无穷大
    mat_id   = -1                       # 默认未命中
    hit_p    = ti.math.vec3(0.0)
    hit_n    = ti.math.vec3(0.0)
    base_col = ti.math.vec3(0.0)

    # ── 检测地面 ───────────────────────────────────────────────────────────
    g_hit, g_t = intersect_ground(ro, rd)
    if g_hit and g_t < best_t:
        best_t = g_t
        mat_id = 0
        p      = ro + g_t * rd          # 计算地面交点
        hit_p  = p
        hit_n  = ti.math.vec3(0.0, 1.0, 0.0)  # 地面法线固定朝上

        # 棋盘格纹理：
        # 将交点的 x、z 坐标向下取整，判断奇偶性
        # 奇偶相同 → 白格，奇偶不同 → 黑格
        cx = int(ti.floor(p.x)) & 1    # x 方向格子奇偶（0 或 1）
        cz = int(ti.floor(p.z)) & 1    # z 方向格子奇偶（0 或 1）
        if (cx ^ cz) == 0:             # XOR：两者相同为白，不同为黑
            base_col = ti.math.vec3(0.9, 0.9, 0.9)   # 浅灰（白格）
        else:
            base_col = ti.math.vec3(0.1, 0.1, 0.1)   # 深灰（黑格）

    # ── 检测红球（漫反射）─────────────────────────────────────────────────
    d_hit, d_t, d_n = intersect_sphere(ro, rd, SPHERE_DIFF_POS, SPHERE_R)
    if d_hit and d_t < best_t:
        best_t   = d_t
        mat_id   = 1
        hit_p    = ro + d_t * rd
        hit_n    = d_n
        base_col = ti.math.vec3(0.85, 0.15, 0.10)   # 红色

    # ── 检测银球（镜面）───────────────────────────────────────────────────
    m_hit, m_t, m_n = intersect_sphere(ro, rd, SPHERE_MIRR_POS, SPHERE_R)
    if m_hit and m_t < best_t:
        best_t   = m_t
        mat_id   = 2
        hit_p    = ro + m_t * rd
        hit_n    = m_n
        base_col = ti.math.vec3(0.8, 0.8, 0.8)      # 银色

    return mat_id, hit_p, hit_n, base_col


# ══════════════════════════════════════════════════════════════════════════════
#  阴影检测
#  从交点向光源发射"暗影射线"，判断路径上是否有遮挡物
# ══════════════════════════════════════════════════════════════════════════════

@ti.func
def in_shadow(hit_p, hit_n, lpos):
    """
    判断交点是否处于阴影中。
    
    ⚠️ Shadow Acne 修复：
       暗影射线起点必须沿法线方向偏移 EPS，
       否则射线会立刻与自身表面相交，产生错误的自阴影噪点。
    
    参数：
        hit_p : 着色点坐标
        hit_n : 着色点法向量
        lpos  : 点光源位置
    返回：
        True  = 处于阴影中
        False = 光源可见
    """
    # 暗影射线起点：沿法线偏移一小段距离，跳过自身表面
    shadow_origin = hit_p + hit_n * EPS

    to_light      = lpos - shadow_origin   # 指向光源的向量
    dist_to_light = to_light.norm()        # 到光源的距离
    shadow_dir    = to_light / dist_to_light  # 归一化方向

    shadowed = False

    # 依次检测三个物体是否遮挡光线
    # 注意：只有交点 t < dist_to_light 才算真正遮挡（光源之前的遮挡）
    gh, gt = intersect_ground(shadow_origin, shadow_dir)
    if gh and gt < dist_to_light:
        shadowed = True

    if not shadowed:
        dh, dt, _ = intersect_sphere(shadow_origin, shadow_dir,
                                     SPHERE_DIFF_POS, SPHERE_R)
        if dh and dt < dist_to_light:
            shadowed = True

    if not shadowed:
        mh, mt, _ = intersect_sphere(shadow_origin, shadow_dir,
                                     SPHERE_MIRR_POS, SPHERE_R)
        if mh and mt < dist_to_light:
            shadowed = True

    return shadowed


# ══════════════════════════════════════════════════════════════════════════════
#  Phong 光照模型
#  计算漫反射材质的着色颜色
# ══════════════════════════════════════════════════════════════════════════════

@ti.func
def phong_diffuse(hit_p, hit_n, base_col, lpos):
    """
    Phong 反射模型 = 环境光 + 漫反射 + 镜面高光
    参数：
        hit_p    : 着色点坐标
        hit_n    : 着色点法向量
        base_col : 物体固有颜色
        lpos     : 光源位置
    返回：
        最终 RGB 颜色
    """
    light_color = ti.math.vec3(1.0, 1.0, 0.95)  # 略带暖黄的白色光

    # ① 环境光（Ambient）：模拟场景中无处不在的间接光
    #    无论阴影与否都存在，防止阴影区域完全漆黑
    ambient = 0.15 * base_col * light_color

    # ② 漫反射（Diffuse）和 ③ 高光（Specular）初始化为 0
    #    若在阴影中，它们保持 0，不参与最终颜色
    diffuse  = ti.math.vec3(0.0)
    specular = ti.math.vec3(0.0)

    # ✅ 只有不在阴影中，才计算漫反射和高光
    if not in_shadow(hit_p, hit_n, lpos):

        # 从着色点指向光源的单位向量
        to_light = (lpos - hit_p).normalized()

        # 近似的观察方向（相机在 +z 方向看向场景）
        cam_dir  = ti.math.vec3(0.0, 0.0, 1.0)

        # Lambert 漫反射：cos(入射角) = dot(法线, 光线方向)
        # ti.max(..., 0) 保证背光面不产生负值颜色
        diff    = ti.max(hit_n.dot(to_light), 0.0)
        diffuse = 0.85 * diff * base_col * light_color

        # Blinn-Phong 高光：使用半程向量（half vector）代替完整反射向量
        # 半程向量 = 归一化(光线方向 + 视线方向)
        half_v   = (to_light + cam_dir).normalized()
        spec     = ti.pow(ti.max(hit_n.dot(half_v), 0.0), 32.0)  # 32 = 高光锐度
        specular = 0.4 * spec * light_color

    # ✅ 唯一的 return，在函数末尾
    return ambient + diffuse + specular


# ══════════════════════════════════════════════════════════════════════════════
#  主渲染 Kernel
#  对每个像素并行执行光线追踪
# ══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def render():
    """
    Taichi kernel：对 pixels 中每个 (px, py) 并行执行。
    
    流程：
      1. 根据像素坐标生成主光线（透视投影）
      2. 迭代弹射（最多 MAX_B 次）：
         - 未命中 → 采样天空颜色，停止
         - 漫反射 → Phong 着色，停止
         - 镜面   → 计算反射方向，继续弹射，throughput *= 0.8
    """
    lpos  = light_pos[None]    # 读取当前光源位置
    max_b = max_bounces[None]  # 读取用户设定的最大弹射次数

    # Taichi 的并行 for：对所有像素并行执行以下代码
    for px, py in pixels:

        # ── Step 1：透视相机，生成主光线 ────────────────────────────────────
        cam_origin = ti.math.vec3(0.0, 0.5, 10.5)   # 相机位于 z=4.5，稍微偏上

        fov    = 0.6                                  # tan(半视角)，控制视野宽窄
        aspect = float(WIDTH) / float(HEIGHT)         # 宽高比，防止画面拉伸

        # 将像素坐标 [0, WIDTH] 映射到 [-fov*aspect/2, +fov*aspect/2]
        uv_x = (float(px) / float(WIDTH)  - 0.5) * fov * aspect
        uv_y = (float(py) / float(HEIGHT) - 0.5) * fov

        # 主光线：从相机出发，朝 -z 方向（即屏幕里面）射出
        ro = cam_origin
        rd = ti.math.vec3(uv_x, uv_y, -1.0).normalized()

        # ── Step 2：迭代光线弹射 ─────────────────────────────────────────────
        throughput = 1.0                      # 光线能量衰减系数，初始为 1（无衰减）
        final_col  = ti.math.vec3(0.0)        # 最终像素颜色，初始为黑色
        done       = False                    # ✅ 用 flag 代替 break（GPU 兼容写法）

        # ✅ for 上界必须是编译期常量 MAX_B
        #    bounce < max_b 在内部动态限制实际执行次数
        for bounce in range(MAX_B):
            if not done and bounce < max_b:

                # 对当前射线做场景求交
                mat_id, hit_p, hit_n, base_col = scene_intersect(ro, rd)

                if mat_id == -1:
                    # —— 未命中任何物体：采样天空渐变色 ——
                    # 根据射线方向的 y 分量在白色和蓝色之间线性插值
                    t_val = 0.5 * (rd.y + 1.0)      # 把 [-1,1] 映射到 [0,1]
                    sky   = (1.0 - t_val) * ti.math.vec3(1.0, 1.0, 1.0) \
                          +  t_val        * ti.math.vec3(0.45, 0.65, 1.0)
                    final_col += throughput * sky
                    done = True                      # 停止弹射

                elif mat_id == 0 or mat_id == 1:
                    # —— 漫反射材质（地面 or 红球）：计算 Phong 着色后停止 ——
                    color      = phong_diffuse(hit_p, hit_n, base_col, lpos)
                    final_col += throughput * color
                    done = True                      # 漫反射不再继续弹射

                else:
                    # —— 镜面材质（银球）：计算反射方向，继续弹射 ——
                    #
                    # 反射向量公式：R = d - 2*(d·n)*n
                    #   d : 入射方向 rd
                    #   n : 表面法向量 hit_n
                    reflect_dir = (rd - 2.0 * rd.dot(hit_n) * hit_n).normalized()

                    # ⚠️ 反射起点也需沿法线偏移 EPS，防止反射射线立刻击中自身（自相交）
                    ro = hit_p + hit_n * EPS
                    rd = reflect_dir

                    # 每次镜面反射损失 20% 能量（反射率 0.8）
                    throughput *= 0.8
                    # 不设 done=True，继续下一次 bounce

        # 将最终颜色写入像素缓冲区，clamp 防止超出 [0,1] 范围
        pixels[px, py] = ti.math.clamp(final_col, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  主程序：窗口 + UI 交互
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 设置初始参数
    light_pos[None]   = [2.0, 4.0, 3.0]   # 光源默认位置：右上前方
    max_bounces[None] = 3                  # 默认弹射次数

    # 创建窗口（Taichi UI）
    window = ti.ui.Window("Ray Tracing Lab - CPU", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()   # 画布，用于显示 pixels field
    gui    = window.get_gui()      # GUI 组件，用于绘制滑块

    # Python 端存储滑块当前值（每帧从 GUI 读取后写入 Taichi field）
    lx, ly, lz = 2.0, 4.0, 3.0
    mb = 3

    while window.running:

        # ── 绘制 UI 控制面板 ──────────────────────────────────────────────
        # sub_window 参数：标题, 左上角x比例, 左上角y比例, 宽度比例, 高度比例
        with gui.sub_window("Controls", 0.02, 0.02, 0.30, 0.38):
            gui.text("=== Light Position ===")
            # slider_float：标签, 当前值, 最小值, 最大值 → 返回新值
            lx = gui.slider_float("Light X", lx, -6.0, 6.0)
            ly = gui.slider_float("Light Y", ly,  0.5, 8.0)
            lz = gui.slider_float("Light Z", lz, -4.0, 8.0)
            gui.text("")
            gui.text("=== Ray Settings ===")
            # slider_int：整数滑块
            mb = gui.slider_int("Max Bounces", mb, 1, 5)
            gui.text("  1=无反射  2+=有反射")

        # ── 将 Python 端的值同步写入 Taichi field ────────────────────────
        # Taichi kernel 只能读取 field，不能直接读 Python 变量
        light_pos[None]   = [lx, ly, lz]
        max_bounces[None] = mb

        # ── 执行渲染（每帧重新计算所有像素）────────────────────────────────
        # CPU 后端每帧渲染较慢，约 0.5~2 秒/帧（取决于机器性能）
        render()

        # ── 将像素数据显示到窗口 ─────────────────────────────────────────
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()