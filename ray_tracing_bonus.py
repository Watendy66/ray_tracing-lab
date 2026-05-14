"""
Whitted-Style Ray Tracing Lab —— 选做扩展版
新增：① 折射/玻璃材质（斯涅尔定律 + 全内反射）
      ② 抗锯齿（MSAA，每像素多次采样取平均）

运行：python ray_tracing_bonus.py
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH  = 800
HEIGHT = 600

# MSAA 采样数（选做②）—— 1 = 关闭, 4/8/16 = 开启
MSAA_SAMPLES = 4

pixels     = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos  = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())
msaa_flag  = ti.field(dtype=ti.i32, shape=())   # 1=开启 MSAA

GROUND_Y        = -1.0
SPHERE_R        = 1.0
SPHERE_GLASS_POS = ti.math.vec3(-1.5, 0.0, 0.0)  # 左球→玻璃
SPHERE_MIRR_POS  = ti.math.vec3( 1.5, 0.0, 0.0)  # 右球→镜面
EPS  = 1e-4
INF  = 1e30
IOR  = 1.5   # 玻璃折射率

# ─── 随机数（简单 hash，用于 MSAA 抖动）────────────────────────────────────
@ti.func
def rand_float(seed: int) -> float:
    """返回 [0,1) 伪随机数"""
    s = (seed ^ 0x12345678) * 0x27d4eb2d
    s = (s ^ (s >> 15)) * 0x27d4eb2d
    s = s ^ (s >> 13)
    return float(s & 0x7FFFFFFF) / float(0x7FFFFFFF)


# ─── 几何求交 ────────────────────────────────────────────────────────────────

@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc   = ro - center
    a    = rd.dot(rd)
    b    = 2.0 * oc.dot(rd)
    c    = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c
    hit  = False
    t    = INF
    n    = ti.math.vec3(0.0, 1.0, 0.0)
    if disc >= 0.0:
        sq = ti.sqrt(disc)
        t0 = (-b - sq) / (2.0 * a)
        t1 = (-b + sq) / (2.0 * a)
        if t0 > EPS:
            t = t0; hit = True
        elif t1 > EPS:
            t = t1; hit = True
        if hit:
            p = ro + t * rd
            n = (p - center).normalized()
    return hit, t, n


@ti.func
def intersect_ground(ro, rd):
    hit = False
    t   = INF
    if ti.abs(rd.y) > 1e-6:
        tc = (GROUND_Y - ro.y) / rd.y
        if tc > EPS:
            hit = True; t = tc
    return hit, t


# ─── 场景求交
# mat_id: 0=地面, 1=玻璃球, 2=镜面球, -1=miss
@ti.func
def scene_intersect(ro, rd):
    best_t   = INF
    mat_id   = -1
    hit_p    = ti.math.vec3(0.0)
    hit_n    = ti.math.vec3(0.0)
    base_col = ti.math.vec3(0.0)

    g_hit, g_t = intersect_ground(ro, rd)
    if g_hit and g_t < best_t:
        best_t = g_t; mat_id = 0
        p      = ro + g_t * rd
        hit_p  = p
        hit_n  = ti.math.vec3(0.0, 1.0, 0.0)
        cx = int(ti.floor(p.x)) & 1
        cz = int(ti.floor(p.z)) & 1
        if (cx ^ cz) == 0:
            base_col = ti.math.vec3(0.9, 0.9, 0.9)
        else:
            base_col = ti.math.vec3(0.1, 0.1, 0.1)

    gl_hit, gl_t, gl_n = intersect_sphere(ro, rd, SPHERE_GLASS_POS, SPHERE_R)
    if gl_hit and gl_t < best_t:
        best_t = gl_t; mat_id = 1
        hit_p  = ro + gl_t * rd; hit_n = gl_n
        base_col = ti.math.vec3(0.9, 0.98, 1.0)   # 淡蓝玻璃

    m_hit, m_t, m_n = intersect_sphere(ro, rd, SPHERE_MIRR_POS, SPHERE_R)
    if m_hit and m_t < best_t:
        best_t = m_t; mat_id = 2
        hit_p  = ro + m_t * rd; hit_n = m_n
        base_col = ti.math.vec3(0.8, 0.8, 0.8)

    return mat_id, hit_p, hit_n, base_col


# ─── 阴影检测 ────────────────────────────────────────────────────────────────

@ti.func
def in_shadow(hit_p, hit_n, lpos):
    so  = hit_p + hit_n * EPS
    tl  = lpos - so
    dtl = tl.norm()
    sd  = tl / dtl
    shadowed = False

    gh, gt = intersect_ground(so, sd)
    if gh and gt < dtl: shadowed = True
    if not shadowed:
        glh, glt, _ = intersect_sphere(so, sd, SPHERE_GLASS_POS, SPHERE_R)
        if glh and glt < dtl: shadowed = True
    if not shadowed:
        mh, mt, _ = intersect_sphere(so, sd, SPHERE_MIRR_POS, SPHERE_R)
        if mh and mt < dtl: shadowed = True
    return shadowed


# ─── Phong 着色 ──────────────────────────────────────────────────────────────

@ti.func
def phong_diffuse(hit_p, hit_n, base_col, lpos):

    light_color = ti.math.vec3(1.0, 1.0, 0.95)

    ambient = 0.15 * base_col * light_color

    diffuse  = ti.math.vec3(0.0)
    specular = ti.math.vec3(0.0)

    if not in_shadow(hit_p, hit_n, lpos):

        tl = (lpos - hit_p).normalized()

        cam = ti.math.vec3(0.0, 0.0, 1.0)

        diff = ti.max(hit_n.dot(tl), 0.0)

        diffuse = (
            0.85 *
            diff *
            base_col *
            light_color
        )

        hv = (tl + cam).normalized()

        spec = ti.pow(
            ti.max(hit_n.dot(hv), 0.0),
            32.0
        )

        specular = 0.4 * spec * light_color

    return ambient + diffuse + specular

# ─── 斯涅尔折射（选做①）────────────────────────────────────────────────────

@ti.func
def refract(rd, n, eta):
    """
    计算折射方向（斯涅尔定律）。
    返回 (refracted_dir, total_internal_reflection: bool)
    """
    cos_i    = -rd.dot(n)
    sin2_t   = eta * eta * (1.0 - cos_i * cos_i)
    tir      = sin2_t > 1.0   # 全内反射
    refr_dir = rd
    if not tir:
        refr_dir = eta * rd + (eta * cos_i - ti.sqrt(1.0 - sin2_t)) * n
        refr_dir = refr_dir.normalized()
    return refr_dir, tir


@ti.func
def schlick(cos_theta, ior):
    """Schlick 近似菲涅尔系数"""
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5


# ─── 单条射线追踪（含折射）────────────────────────────────────────────────────

@ti.func
def trace_ray(ro, rd, lpos, max_b):

    throughput = 1.0
    final_col  = ti.math.vec3(0.0)

    done = False

    for bounce in range(5):

        if not done and bounce < max_b:

            mat_id, hit_p, hit_n, base_col = scene_intersect(ro, rd)

            # ─────────────────────────────
            # Sky
            # ─────────────────────────────
            if mat_id == -1:

                t_val = 0.5 * (rd.y + 1.0)

                sky = (
                    (1.0 - t_val)
                    * ti.math.vec3(1.0, 1.0, 1.0)
                    +
                    t_val
                    * ti.math.vec3(0.45, 0.65, 1.0)
                )

                final_col += throughput * sky

                done = True

            # ─────────────────────────────
            # Ground
            # ─────────────────────────────
            elif mat_id == 0:

                final_col += (
                    throughput *
                    phong_diffuse(
                        hit_p,
                        hit_n,
                        base_col,
                        lpos
                    )
                )

                done = True

            # ─────────────────────────────
            # Glass
            # ─────────────────────────────
            elif mat_id == 1:

                inside = rd.dot(hit_n) > 0.0

                normal = -hit_n if inside else hit_n

                eta = IOR if inside else (1.0 / IOR)

                cos_in = ti.abs(rd.dot(normal))

                fresnel = schlick(cos_in, IOR)

                refr_dir, tir = refract(
                    rd,
                    normal,
                    eta
                )

                # 全内反射
                if tir or fresnel > 0.9:

                    ro = hit_p + normal * EPS

                    rd = (
                        rd
                        -
                        2.0 * rd.dot(normal) * normal
                    ).normalized()

                    throughput *= fresnel

                # 折射
                else:

                    ro = hit_p - normal * EPS

                    rd = refr_dir

                    throughput *= (
                        1.0 - fresnel
                    ) * 0.95

            # ─────────────────────────────
            # Mirror
            # ─────────────────────────────
            elif mat_id == 2:

                ro = hit_p + hit_n * EPS

                rd = (
                    rd
                    -
                    2.0 * rd.dot(hit_n) * hit_n
                ).normalized()

                throughput *= 0.8

    return ti.math.clamp(final_col, 0.0, 1.0)


# ─── 主渲染 Kernel（MSAA）────────────────────────────────────────────────────

@ti.kernel
def render():
    lpos  = light_pos[None]
    max_b = max_bounces[None]
    use_msaa = msaa_flag[None]

    for px, py in pixels:
        cam_origin = ti.math.vec3(0.0, 0.5, 10.5)
        fov        = 0.6
        aspect     = float(WIDTH) / float(HEIGHT)

        accum = ti.math.vec3(0.0)
        n_samples = MSAA_SAMPLES if use_msaa == 1 else 1

        for s in range(MSAA_SAMPLES):   # 静态展开，用 if 控制有效采样数
            if s < n_samples:
                seed_x = px * 7919 + py * 104729 + s * 2053
                seed_y = px * 6271 + py *  98317 + s * 3571
                jx = 0.0
                jy = 0.0

                if use_msaa == 1:
                    jx = rand_float(seed_x) - 0.5
                    jy = rand_float(seed_y) - 0.5
            

                uv_x = ((float(px) + jx) / float(WIDTH)  - 0.5) * fov * aspect
                uv_y = ((float(py) + jy) / float(HEIGHT) - 0.5) * fov
                rd   = ti.math.vec3(uv_x, uv_y, -1.0).normalized()

                accum += trace_ray(cam_origin, rd, lpos, max_b)

        pixels[px, py] = accum / float(n_samples)


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    light_pos[None]   = [2.0, 4.0, 3.0]
    max_bounces[None] = 3
    msaa_flag[None]   = 0

    window = ti.ui.Window("Ray Tracing Lab (Bonus)", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui    = window.get_gui()

    lx, ly, lz = 2.0, 4.0, 3.0
    mb   = 3
    msaa = False

    while window.running:
        with gui.sub_window("Controls", 0.02, 0.02, 0.32, 0.44):
            gui.text("=== Light Position ===")
            lx   = gui.slider_float("Light X", lx, -6.0,  6.0)
            ly   = gui.slider_float("Light Y", ly,  0.5,  8.0)
            lz   = gui.slider_float("Light Z", lz, -4.0,  8.0)
            gui.text("")
            gui.text("=== Ray Settings ===")
            mb   = gui.slider_int("Max Bounces", mb, 1, 5)
            gui.text("")
            gui.text("=== Anti-Aliasing ===")
            msaa = gui.checkbox("MSAA x4", msaa)
            gui.text("(MSAA 会降低帧率)")

        light_pos[None]   = [lx, ly, lz]
        max_bounces[None] = mb
        msaa_flag[None]   = 1 if msaa else 0

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
