"""
APP 1: Mặt cong, đạo hàm riêng, tiếp tuyến và mặt phẳng tiếp diện
Cài đặt:  pip install numpy sympy matplotlib
Chạy:     python app1_dao_ham_rieng_hinh_hoc.py

- Ô nhập hàm f(x,y): gõ kiểu x^2 + y^2, sin(x)*cos(y), exp(-(x^2+y^2)), ...
- Kéo thanh trượt HOẶC gõ số vào ô "nhập x0", "nhập y0" rồi nhấn Enter để đổi điểm M0(x0, y0, f(x0,y0)).
  (giá trị nhập được kẹp trong [-L, L], mặc định L = 3)
- Đỏ : lát cắt y = y0 và tiếp tuyến có hệ số góc f_x  (đạo hàm riêng theo x)
- Xanh: lát cắt x = x0 và tiếp tuyến có hệ số góc f_y  (đạo hàm riêng theo y)
- Cam : mặt phẳng tiếp diện  z = f0 + f_x(x-x0) + f_y(y-y0)
- Tím : vectơ pháp tuyến n = (f_x, f_y, -1)
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application, convert_xor)

x, y = sp.symbols("x y")
TR = standard_transformations + (implicit_multiplication_application, convert_xor)
L = 3.0          # miền vẽ: [-L, L] x [-L, L]
H = 1.0          # nửa kích thước mảnh mặt phẳng tiếp diện / đoạn tiếp tuyến

state = {}


def set_function(text):
    expr = parse_expr(text, local_dict={"x": x, "y": y}, transformations=TR)
    state["expr"] = expr
    state["f"] = sp.lambdify((x, y), expr, "numpy")
    state["fx"] = sp.lambdify((x, y), sp.diff(expr, x), "numpy")
    state["fy"] = sp.lambdify((x, y), sp.diff(expr, y), "numpy")
    state["fx_s"] = sp.diff(expr, x)
    state["fy_s"] = sp.diff(expr, y)


def F(fn, X, Y):
    """Gọi hàm lambdify và ép về mảng cùng kích thước (phòng hàm hằng)."""
    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    return np.asarray(fn(X, Y), float) * np.ones_like(X)


def draw(_=None):
    ax = state["ax"]
    elev, azim = ax.elev, ax.azim
    ax.clear()
    x0, y0 = state["sx"].val, state["sy"].val

    g = np.linspace(-L, L, 70)
    X, Y = np.meshgrid(g, g)
    Z = F(state["f"], X, Y)
    Z = np.where(np.isfinite(Z), Z, np.nan)
    lo, hi = np.nanpercentile(Z, [1, 99])
    pad = 0.15 * (hi - lo + 1e-9)
    Zc = np.clip(Z, lo - pad, hi + pad)

    ax.plot_surface(X, Y, Zc, cmap="viridis", alpha=0.45, linewidth=0, antialiased=True)

    z0 = float(F(state["f"], x0, y0))
    a = float(F(state["fx"], x0, y0))   # f_x(x0,y0)
    b = float(F(state["fy"], x0, y0))   # f_y(x0,y0)

    # mặt phẳng tiếp diện
    gx = np.linspace(x0 - H, x0 + H, 8)
    gy = np.linspace(y0 - H, y0 + H, 8)
    XP, YP = np.meshgrid(gx, gy)
    ZP = z0 + a * (XP - x0) + b * (YP - y0)
    ax.plot_surface(XP, YP, ZP, color="orange", alpha=0.65, linewidth=0)

    # lát cắt
    t = np.linspace(-L, L, 200)
    ax.plot(t, np.full_like(t, y0), np.clip(F(state["f"], t, np.full_like(t, y0)), lo - pad, hi + pad),
            "r-", lw=2, label="lát cắt y = y0")
    ax.plot(np.full_like(t, x0), t, np.clip(F(state["f"], np.full_like(t, x0), t), lo - pad, hi + pad),
            "b-", lw=2, label="lát cắt x = x0")

    # tiếp tuyến
    s = np.array([-H, H])
    ax.plot(x0 + s, [y0, y0], z0 + a * s, "r--", lw=3, label=f"tiếp tuyến theo x (f_x = {a:.3f})")
    ax.plot([x0, x0], y0 + s, z0 + b * s, "b--", lw=3, label=f"tiếp tuyến theo y (f_y = {b:.3f})")

    ax.scatter([x0], [y0], [z0], color="k", s=60, zorder=10)

    # vectơ pháp tuyến n = (f_x, f_y, -1) của mặt z = f(x,y) tại M0 (vẽ chuẩn hoá cho dễ nhìn)
    ax.quiver(x0, y0, z0, a, b, -1.0, length=1.5, normalize=True, color="m", linewidth=3)

    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(lo - pad, hi + pad)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"z = {sp.sstr(state['expr'])}   |   M0 = ({x0:.2f}, {y0:.2f}, {z0:.2f})\n"
        f"Tiếp diện: z = {z0:.3f} + ({a:.3f})(x - {x0:.2f}) + ({b:.3f})(y - {y0:.2f})\n"
        f"Vectơ pháp tuyến (tím): n = (f_x, f_y, -1) = ({a:.3f}, {b:.3f}, -1)",
        fontsize=10)
    state["fig"].canvas.draw_idle()


def on_num(which):
    """Nhập số vào ô x0 hoặc y0 -> cập nhật thanh trượt (kẹp trong [-L, L])."""
    def handler(text):
        if state.get("lock"):
            return
        try:
            v = float(text.replace(",", "."))
        except ValueError:
            print("Hãy nhập một số, ví dụ 1.5")
            return
        v = max(-L, min(L, v))
        state["lock"] = True
        try:
            state[which].set_val(v)          # gọi draw() qua thanh trượt
        finally:
            state["lock"] = False
        sync_boxes()
    return handler


def sync_boxes():
    """Đồng bộ ô nhập với giá trị thanh trượt."""
    state["lock"] = True
    try:
        state["tx"].set_val(f"{state['sx'].val:.3f}")
        state["ty"].set_val(f"{state['sy'].val:.3f}")
    finally:
        state["lock"] = False


def on_slider(_=None):
    if not state.get("lock"):
        sync_boxes()
    draw()


def on_text(text):
    try:
        set_function(text)
        draw()
    except Exception as e:
        print("Hàm không hợp lệ:", e)


def main():
    set_function("sin(x)*cos(y) + 0.1*x^2")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.05, 0.22, 0.9, 0.70], projection="3d")
    state.update(fig=fig, ax=ax)
    state["sx"] = Slider(fig.add_axes([0.15, 0.12, 0.5, 0.03]), "x0", -L, L, valinit=0.5)
    state["sy"] = Slider(fig.add_axes([0.15, 0.07, 0.5, 0.03]), "y0", -L, L, valinit=0.5)
    state["tx"] = TextBox(fig.add_axes([0.76, 0.12, 0.14, 0.03]), "nhập x0 ", initial="0.500")
    state["ty"] = TextBox(fig.add_axes([0.76, 0.07, 0.14, 0.03]), "nhập y0 ", initial="0.500")
    tb = TextBox(fig.add_axes([0.2, 0.01, 0.6, 0.04]), "f(x,y) = ", initial="sin(x)*cos(y) + 0.1*x^2")
    state["tb"] = tb
    state["sx"].on_changed(on_slider)
    state["sy"].on_changed(on_slider)
    state["tx"].on_submit(on_num("sx"))
    state["ty"].on_submit(on_num("sy"))
    tb.on_submit(on_text)
    draw()
    plt.show()


if __name__ == "__main__":
    main()