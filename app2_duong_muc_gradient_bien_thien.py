"""
APP 2: Đọc sự biến thiên của hàm số trên bản đồ đường mức
       Xét dấu f_x, f_y, f_xx, f_xy, f_yy và so sánh đạo hàm riêng cấp 1 tại hai điểm A, B
Cài đặt:  pip install numpy sympy matplotlib
Chạy:     python app2_duong_muc_gradient_bien_thien.py

Bên trái : bản đồ đường mức, trường gradient, điểm A (đỏ), B (xanh), gradient tại A, B
           (vuông góc đường mức), hướng u (tím) tại A, các điểm tới hạn (cực trị, yên ngựa).
Bên phải : lát cắt theo x (qua A, B) và theo y (qua A, B) để thấy dấu và độ dốc f_x, f_y.
Bên dưới : bảng giá trị f, f_x, f_y, f_xx, f_xy, f_yy tại A và B kèm dấu / so sánh.
Mẹo đọc bản đồ đường mức:
  * Đi theo chiều x tăng mà cắt qua đường mức có giá trị lớn dần  => f_x > 0 (ngược lại f_x < 0).
  * Đường mức càng DÀY (sát nhau) => |gradient| càng LỚN => f biến thiên nhanh.
  * Dọc theo chiều x, khoảng cách giữa các đường mức (cùng bước) càng THƯA dần => f_x giảm dần (f_xx < 0)
    và càng DÀY dần => f_x tăng dần (f_xx > 0). Tương tự cho y với f_yy.
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application, convert_xor)

x, y = sp.symbols("x y")
TR = standard_transformations + (implicit_multiplication_application, convert_xor)
L = 3.0
state = {}


def F(fn, X, Y):
    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    return np.asarray(fn(X, Y), float) * np.ones_like(X)


def find_critical_points(expr):
    """Tìm điểm tới hạn (f_x = f_y = 0) bằng nsolve từ lưới điểm xuất phát, phân loại bằng Hessian."""
    fxe, fye = sp.diff(expr, x), sp.diff(expr, y)
    lfx = sp.lambdify((x, y), fxe, "numpy"); lfy = sp.lambdify((x, y), fye, "numpy")
    lxx = sp.lambdify((x, y), sp.diff(fxe, x), "numpy")
    lyy = sp.lambdify((x, y), sp.diff(fye, y), "numpy")
    lxy = sp.lambdify((x, y), sp.diff(fxe, y), "numpy")
    pts = []
    for sx in np.linspace(-L, L, 7):
        for sy in np.linspace(-L, L, 7):
            try:
                sol = sp.nsolve((fxe, fye), (x, y), (sx, sy), tol=1e-10, maxsteps=40)
                px, py = float(sol[0]), float(sol[1])
            except Exception:
                continue
            if abs(px) > L or abs(py) > L:
                continue
            if abs(float(lfx(px, py))) > 1e-6 or abs(float(lfy(px, py))) > 1e-6:
                continue
            if any(abs(px - q[0]) < 1e-3 and abs(py - q[1]) < 1e-3 for q in pts):
                continue
            pts.append((px, py))
    out = []
    for px, py in pts:
        A_, B_, C_ = float(lxx(px, py)), float(lxy(px, py)), float(lyy(px, py))
        D = A_ * C_ - B_ * B_
        if D > 1e-9:
            kind = "cực tiểu" if A_ > 0 else "cực đại"
        elif D < -1e-9:
            kind = "điểm yên ngựa"
        else:
            kind = "chưa kết luận (D=0)"
        out.append((px, py, kind))
    return out


def set_function(text):
    expr = parse_expr(text, local_dict={"x": x, "y": y}, transformations=TR)
    state["expr"] = expr
    d = {"f": expr, "fx": sp.diff(expr, x), "fy": sp.diff(expr, y)}
    d["fxx"] = sp.diff(d["fx"], x); d["fxy"] = sp.diff(d["fx"], y); d["fyy"] = sp.diff(d["fy"], y)
    state["fn"] = {k: sp.lambdify((x, y), v, "numpy") for k, v in d.items()}
    state["crit"] = find_critical_points(expr)


def vals(px, py):
    return {k: float(F(fn, px, py)) for k, fn in state["fn"].items()}


def sg(v, tol=1e-6):
    return "0" if abs(v) < tol else ("+" if v > 0 else "-")


def cmp_abs(a, b, name):
    if abs(abs(a) - abs(b)) < 1e-6:
        return f"|{name}(A)| = |{name}(B)|"
    return f"|{name}(A)| {'>' if abs(a) > abs(b) else '<'} |{name}(B)|"


def draw(_=None):
    ax1, ax2, ax3 = state["ax1"], state["ax2"], state["ax3"]
    for a_ in (ax1, ax2, ax3):
        a_.clear()
    xa, ya = state["sxa"].val, state["sya"].val
    xb, yb = state["sxb"].val, state["syb"].val
    th = np.radians(state["st"].val)
    VA, VB = vals(xa, ya), vals(xb, yb)

    # ---------- bản đồ đường mức ----------
    g = np.linspace(-L, L, 200)
    X, Y = np.meshgrid(g, g)
    Z = F(state["fn"]["f"], X, Y)
    ax1.contourf(X, Y, Z, levels=25, cmap="viridis", alpha=0.75)
    ax1.contour(X, Y, Z, levels=25, colors="k", linewidths=0.4, alpha=0.5)

    gq = np.linspace(-L, L, 15)
    XQ, YQ = np.meshgrid(gq, gq)
    GX, GY = F(state["fn"]["fx"], XQ, YQ), F(state["fn"]["fy"], XQ, YQ)
    N = np.hypot(GX, GY); N[N == 0] = 1
    ax1.quiver(XQ, YQ, GX / N, GY / N, color="white", alpha=0.5, scale=30, width=0.003)

    for (px, py, V, col, name) in ((xa, ya, VA, "red", "A"), (xb, yb, VB, "cyan", "B")):
        try:
            ax1.contour(X, Y, Z, levels=[V["f"]], colors=col, linewidths=2.5)
        except Exception:
            pass
        ax1.plot([px], [py], "o", color=col, ms=9, mec="k")
        ax1.annotate(name, (px, py), textcoords="offset points", xytext=(7, 7), color=col,
                     fontsize=13, weight="bold", bbox=dict(boxstyle="round", fc="black", alpha=0.5))
        gn = np.hypot(V["fx"], V["fy"])
        if gn > 1e-9:
            ax1.annotate("", xy=(px + V["fx"] / gn, py + V["fy"] / gn), xytext=(px, py),
                         arrowprops=dict(arrowstyle="-|>", color="yellow", lw=3))
    u = np.array([np.cos(th), np.sin(th)])
    ax1.annotate("", xy=(xa + u[0], ya + u[1]), xytext=(xa, ya),
                 arrowprops=dict(arrowstyle="-|>", color="magenta", lw=2))
    Du = VA["fx"] * u[0] + VA["fy"] * u[1]

    for px, py, kind in state["crit"]:
        ax1.plot(px, py, "w*", ms=14, mec="k")
        ax1.annotate(kind, (px, py), textcoords="offset points", xytext=(6, -14), color="white",
                     fontsize=8, bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_aspect("equal")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_title("Đường mức; ∇f (vàng) vuông góc đường mức; u (tím) tại A; đường mức qua A (đỏ), qua B (xanh)",
                  fontsize=8.5)

    # ---------- lát cắt ----------
    t = np.linspace(-L, L, 300)
    lo, hi = np.nanpercentile(Z, [1, 99]); pad = 0.2 * (hi - lo + 1e-9)
    f_ = state["fn"]["f"]
    s = np.array([-0.7, 0.7])
    for (px, py, V, col, name) in ((xa, ya, VA, "red", "A"), (xb, yb, VB, "tab:blue", "B")):
        ax2.plot(t, F(f_, t, np.full_like(t, py)), "-", color=col, label=f"f(x, y={py:.2f}) qua {name}")
        ax2.plot([px], [V["f"]], "o", color=col); ax2.plot(px + s, V["f"] + V["fx"] * s, "--", color=col)
        ax3.plot(t, F(f_, np.full_like(t, px), t), "-", color=col, label=f"f(x={px:.2f}, y) qua {name}")
        ax3.plot([py], [V["f"]], "o", color=col); ax3.plot(py + s, V["f"] + V["fy"] * s, "--", color=col)
    for a_, xl in ((ax2, "x"), (ax3, "y")):
        a_.set_ylim(lo - pad, hi + pad); a_.set_xlim(-L, L); a_.grid(alpha=0.3)
        a_.legend(fontsize=7, loc="best"); a_.set_xlabel(xl); a_.set_ylabel("z")
    ax2.set_title("Lát cắt theo x: độ dốc tiếp tuyến = f_x", fontsize=9)
    ax3.set_title("Lát cắt theo y: độ dốc tiếp tuyến = f_y", fontsize=9)

    # ---------- bảng ----------
    def row(name, V):
        return (f"{name}  {V['f']:8.3f} | {V['fx']:8.3f} ({sg(V['fx'])}) | {V['fy']:8.3f} ({sg(V['fy'])}) | "
                f"{V['fxx']:8.3f} ({sg(V['fxx'])}) | {V['fxy']:8.3f} ({sg(V['fxy'])}) | "
                f"{V['fyy']:8.3f} ({sg(V['fyy'])}) | {np.hypot(V['fx'], V['fy']):7.3f}")
    head = "     f(M)    | f_x          | f_y          | f_xx         | f_xy         | f_yy         | |grad f|"
    cmp_line = (f"So sánh cấp 1:  {cmp_abs(VA['fx'], VB['fx'], 'f_x')};  {cmp_abs(VA['fy'], VB['fy'], 'f_y')};  "
                f"{cmp_abs(np.hypot(VA['fx'], VA['fy']), np.hypot(VB['fx'], VB['fy']), 'grad f')}"
                f"  (đường mức dày hơn ở điểm có |grad f| lớn hơn)")
    sign_line = (f"Ý nghĩa dấu tại A:  f_x {sg(VA['fx'])} => f {'tăng' if VA['fx']>0 else 'giảm' if VA['fx']<0 else 'đứng'} khi x tăng;  "
                 f"f_y {sg(VA['fy'])} => f {'tăng' if VA['fy']>0 else 'giảm' if VA['fy']<0 else 'đứng'} khi y tăng;  "
                 f"f_xx {sg(VA['fxx'])} => f_x {'tăng' if VA['fxx']>0 else 'giảm' if VA['fxx']<0 else 'không đổi'} theo x;  "
                 f"D_u f(A) = {Du:.3f} (θ={np.degrees(th):.0f}°)")
    state["info"].set_text("\n".join(["Bảng giá trị (dấu trong ngoặc):", head, row("A", VA), row("B", VB), cmp_line, sign_line]))
    state["fig"].canvas.draw_idle()


def on_text(text):
    try:
        set_function(text)
        draw()
    except Exception as e:
        print("Hàm không hợp lệ:", e)


def main():
    init = "x^2 - y^2 + 0.3*x^3"
    set_function(init)
    fig = plt.figure(figsize=(14, 9))
    state["fig"] = fig
    state["ax1"] = fig.add_axes([0.04, 0.42, 0.40, 0.54])
    state["ax2"] = fig.add_axes([0.54, 0.70, 0.43, 0.24])
    state["ax3"] = fig.add_axes([0.54, 0.43, 0.43, 0.20])

    def mk(rect, label, v, lo=-L, hi=L):
        return Slider(fig.add_axes(rect), label, lo, hi, valinit=v)
    state["sxa"] = mk([0.08, 0.35, 0.3, 0.02], "x_A", 1.0)
    state["sya"] = mk([0.08, 0.32, 0.3, 0.02], "y_A", 1.0)
    state["sxb"] = mk([0.58, 0.35, 0.3, 0.02], "x_B", -1.5)
    state["syb"] = mk([0.58, 0.32, 0.3, 0.02], "y_B", -0.5)
    state["st"] = mk([0.08, 0.29, 0.3, 0.02], "θ(A) °", 45, 0, 360)
    state["tb"] = TextBox(fig.add_axes([0.58, 0.27, 0.35, 0.03]), "f(x,y) = ", initial=init)
    state["info"] = fig.text(0.03, 0.02, "", ha="left", va="bottom", fontsize=8.5, family="monospace")
    for k in ("sxa", "sya", "sxb", "syb", "st"):
        state[k].on_changed(draw)
    state["tb"].on_submit(on_text)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
