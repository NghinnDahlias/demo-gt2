"""
Ứng dụng học đạo hàm riêng và ý nghĩa hình học
Cài đặt: pip install streamlit plotly numpy sympy matplotlib pandas
Chạy: streamlit run web_app.py
"""
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)

st.set_page_config(page_title="Đạo hàm riêng & hình học", layout="wide")
x, y = sp.symbols("x y")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application, convert_xor
)
LIMIT = 3.0

# ------------------------------ Xử lý hàm số
@st.cache_data(show_spinner=False)
def compile_function(source):
    expr = parse_expr(source, local_dict={"x": x, "y": y},
                      transformations=TRANSFORMS)
    derivs = {
        "f": expr,
        "fx": sp.diff(expr, x),
        "fy": sp.diff(expr, y),
    }
    derivs["fxx"] = sp.diff(derivs["fx"], x)
    derivs["fxy"] = sp.diff(derivs["fx"], y)
    derivs["fyx"] = sp.diff(derivs["fy"], x)
    derivs["fyy"] = sp.diff(derivs["fy"], y)
    return expr, derivs


def lambdify_all(derivs):
    return {key: sp.lambdify((x, y), value, "numpy")
            for key, value in derivs.items()}


def evaluate(fn, X, Y):
    X, Y = np.broadcast_arrays(np.asarray(X, dtype=float),
                               np.asarray(Y, dtype=float))
    with np.errstate(all="ignore"):
        result = np.asarray(fn(X, Y), dtype=float)
    return np.broadcast_to(result, X.shape)


def point_values(functions, px, py):
    result = {}
    for key, fn in functions.items():
        value = evaluate(fn, px, py)
        result[key] = float(np.asarray(value).reshape(-1)[0])
    return result


def sign_text(value, eps=1e-8):
    if abs(value) < eps:
        return "0"
    return "+" if value > 0 else "−"


def number_control(label, key, default, low=-LIMIT, high=LIMIT):
    left, right = st.columns([3, 1.25])
    slider_key, number_key = key + "_slider", key + "_number"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(default)
    if number_key not in st.session_state:
        st.session_state[number_key] = float(default)

    def slider_changed():
        st.session_state[number_key] = st.session_state[slider_key]

    def number_changed():
        st.session_state[slider_key] = st.session_state[number_key]

    left.slider(label, low, high, step=0.01, key=slider_key,
                on_change=slider_changed)
    right.number_input("Giá trị", min_value=float(low), max_value=float(high),
                       step=0.05, format="%.2f", key=number_key,
                       on_change=number_changed)
    return float(st.session_state[slider_key])


def safe_grid(function, X, Y):
    Z = evaluate(function, X, Y)
    return np.where(np.isfinite(Z), Z, np.nan)


# ------------------------------ Đầu trang
st.title("Đạo hàm riêng và ý nghĩa hình học")
st.caption("Khám phá đạo hàm riêng, tiếp tuyến – tiếp diện và sự biến thiên của hàm hai biến.")

with st.container(border=True):
    ftext = st.text_input(
        "Nhập hàm số f(x, y)",
        value="x^2 - y^2",
        help="Ví dụ: x^2 + y^2, sin(x)*cos(y), x^2 - y^2"
    )
    st.caption("Có thể dùng ^ để nhập lũy thừa; dùng * cho phép nhân.")
try:
    expr, derivatives = compile_function(ftext)
    functions = lambdify_all(derivatives)
    st.latex(r"z=f(x,y)=" + sp.latex(expr))
except Exception as error:
    st.error(f"Không đọc được hàm số. Hãy kiểm tra biểu thức. Chi tiết: {error}")
    st.stop()

tab_calc, tab_geometry, tab_change, tab_app = st.tabs([
    "1 · TÍNH ĐẠO HÀM RIÊNG",
    "2 · Ý NGHĨA HÌNH HỌC",
    "3 · SỰ BIẾN THIÊN",
    "4 · ỨNG DỤNG"
])
# ================================================================
# TAB 1: TÍNH ĐẠO HÀM RIÊNG
with tab_calc:
    st.subheader("Từ hàm số đến các đạo hàm")
    st.write("Chọn điểm M₀ để tính giá trị các đạo hàm tại điểm đó.")
    c1, c2 = st.columns(2)
    with c1:
        x0 = number_control("Hoành độ x₀", "calc_x0", 0.5)
    with c2:
        y0 = number_control("Tung độ y₀", "calc_y0", 0.5)

    values0 = point_values(functions, x0, y0)
    if not all(np.isfinite(v) for v in values0.values()):
        st.warning("Hàm số hoặc đạo hàm không xác định tại M₀ này.")
    else:
        st.markdown(f"**Điểm đang xét:** M₀ = ({x0:.3f}; {y0:.3f})")
        st.markdown("#### Đạo hàm riêng cấp một")
        a, b = st.columns(2)
        with a:
            st.markdown("**Theo biến x** (giữ y không đổi)")
            st.latex(r"f_x(x,y)=\frac{\partial f}{\partial x}")
            st.latex(r"f_x(x,y)=" + sp.latex(derivatives["fx"]))
            st.latex(
                rf"f_x(M_0)=f_x({x0:.2f},{y0:.2f})={values0['fx']:.5f}"
            )
        with b:
            st.markdown("**Theo biến y** (giữ x không đổi)")
            st.latex(r"f_y(x,y)=\frac{\partial f}{\partial y}")
            st.latex(r"f_y(x,y)=" + sp.latex(derivatives["fy"]))
            st.latex(
                rf"f_y(M_0)=f_y({x0:.2f},{y0:.2f})={values0['fy']:.5f}"
            )

        st.info(
            f"Tại M₀: khi tăng x một lượng rất nhỏ và giữ y cố định, "
            f"z thay đổi xấp xỉ {values0['fx']:.4f} lần lượng tăng của x. "
            f"Tương tự, khi tăng y và giữ x cố định, z thay đổi xấp xỉ "
            f"{values0['fy']:.4f} lần lượng tăng của y."
        )

        st.markdown("#### Đạo hàm riêng cấp hai")
        st.caption("Tiếp tục lấy đạo hàm các đạo hàm riêng cấp một.")
        second_cols = st.columns(4)
        second_items = [
            ("fₓₓ", "fxx", r"\frac{\partial^2 f}{\partial x^2}"),
            ("fₓᵧ", "fxy", r"\frac{\partial^2 f}{\partial y\,\partial x}"),
            ("fᵧₓ", "fyx", r"\frac{\partial^2 f}{\partial x\,\partial y}"),
            ("fᵧᵧ", "fyy", r"\frac{\partial^2 f}{\partial y^2}"),
        ]
        for col, (label, key, formula) in zip(second_cols, second_items):
            with col:
                st.markdown(f"**{label}**")
                st.latex(formula)
                st.latex(sp.latex(derivatives[key]))
                st.metric("Tại M₀", f"{values0[key]:.4f}")

        with st.expander("Xem cách tính từng bước"):
            st.markdown("**Quy tắc:** Khi tính đạo hàm riêng theo x, xem y là hằng số; "
                        "khi tính theo y, xem x là hằng số.")
            st.markdown("**Tính fₓ:**")
            st.latex(r"f_x=\frac{\partial}{\partial x}\left(" +
                     sp.latex(expr) + r"\right)=" + sp.latex(derivatives["fx"]))
            st.markdown("**Tính fᵧ:**")
            st.latex(r"f_y=\frac{\partial}{\partial y}\left(" +
                     sp.latex(expr) + r"\right)=" + sp.latex(derivatives["fy"]))
            st.caption("Ứng dụng SymPy để tính và rút gọn biểu thức. Phần này hiển thị biểu thức trước và sau phép lấy đạo hàm; chưa diễn giải từng phép biến đổi đại số trung gian.")

        st.markdown("#### Bảng tổng hợp tại M₀")
        summary = pd.DataFrame([
            {"Đại lượng": key, "Biểu thức": str(derivatives[key]),
             "Giá trị tại M₀": round(values0[key], 6)}
            for key in ["f", "fx", "fy", "fxx", "fxy", "fyx", "fyy"]
        ])
        st.dataframe(summary, hide_index=True, use_container_width=True)

# ================================================================
# TAB 2: HÌNH HỌC
with tab_geometry:
    st.subheader("Mặt cong, lát cắt, tiếp tuyến và tiếp diện")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Chọn điểm M₀ trên mặt cong**")
        gx0 = number_control("x₀", "geo_x0", 0.5)
        gy0 = number_control("y₀", "geo_y0", 0.5)
        V = point_values(functions, gx0, gy0)
        z0, fx0, fy0 = V["f"], V["fx"], V["fy"]
        st.markdown(f"**M₀** = ({gx0:.2f}, {gy0:.2f}, {z0:.3f})")
        st.markdown("**Đạo hàm riêng tại M₀**")
        st.latex(rf"f_x(M_0)={fx0:.4f},\quad f_y(M_0)={fy0:.4f}")
        st.markdown("**Phương trình tiếp diện**")
        st.latex(
            rf"z={z0:.3f}+({fx0:.3f})(x-{gx0:.2f})"
            rf"+({fy0:.3f})(y-{gy0:.2f})"
        )
        st.markdown("**Một vectơ pháp tuyến**")
        st.latex(
            rf"\vec n=(f_x,f_y,-1)=({fx0:.3f},{fy0:.3f},-1)"
        )
        with st.expander("Giải thích màu sắc"):
            st.markdown(
                "- Mặt trong: đồ thị z = f(x,y).\n"
                "- Đỏ: lát cắt y = y₀ và tiếp tuyến theo x.\n"
                "- Xanh: lát cắt x = x₀ và tiếp tuyến theo y.\n"
                "- Cam: tiếp diện tại M₀.\n"
                "- Tím: vectơ pháp tuyến."
            )

    xx = np.linspace(-LIMIT, LIMIT, 65)
    XX, YY = np.meshgrid(xx, xx)
    ZZ = safe_grid(functions["f"], XX, YY)
    finite = ZZ[np.isfinite(ZZ)]
    if finite.size == 0:
        st.warning("Không thể vẽ mặt cong trong miền đang chọn.")
    else:
        zlo, zhi = np.nanpercentile(finite, [2, 98])
        pad = max((zhi - zlo) * 0.15, 0.2)
        ZZ = np.clip(ZZ, zlo - pad, zhi + pad)
        fig = go.Figure()
        fig.add_surface(x=XX, y=YY, z=ZZ, colorscale="Viridis",
                        opacity=0.72, showscale=False, name="z=f(x,y)")
        h = 0.85
        tx = np.linspace(gx0 - h, gx0 + h, 2)
        ty = np.linspace(gy0 - h, gy0 + h, 2)
        TX, TY = np.meshgrid(tx, ty)
        TZ = z0 + fx0 * (TX - gx0) + fy0 * (TY - gy0)
        fig.add_surface(x=TX, y=TY, z=TZ, opacity=0.55,
                        colorscale=[[0, "orange"], [1, "orange"]],
                        showscale=False, name="Tiếp diện")
        t = np.linspace(-LIMIT, LIMIT, 180)
        z_cut_x = safe_grid(functions["f"], t, np.full_like(t, gy0))
        z_cut_y = safe_grid(functions["f"], np.full_like(t, gx0), t)
        fig.add_scatter3d(x=t, y=np.full_like(t, gy0), z=z_cut_x,
                          mode="lines", line=dict(color="red", width=5),
                          name="Lát cắt y=y₀")
        fig.add_scatter3d(x=np.full_like(t, gx0), y=t, z=z_cut_y,
                          mode="lines", line=dict(color="deepskyblue", width=5),
                          name="Lát cắt x=x₀")
        s = np.array([-h, h])
        fig.add_scatter3d(
            x=gx0+s, y=[gy0, gy0], z=z0+fx0*s, mode="lines",
            line=dict(color="red", width=8, dash="dash"),
            name="Tiếp tuyến theo x"
        )
        fig.add_scatter3d(
            x=[gx0, gx0], y=gy0+s, z=z0+fy0*s, mode="lines",
            line=dict(color="deepskyblue", width=8, dash="dash"),
            name="Tiếp tuyến theo y"
        )
        normal = np.array([fx0, fy0, -1.0])
        normal = normal / np.linalg.norm(normal)
        scale = 1.1
        fig.add_scatter3d(
            x=[gx0, gx0+scale*normal[0]],
            y=[gy0, gy0+scale*normal[1]],
            z=[z0, z0+scale*normal[2]], mode="lines+markers",
            line=dict(color="magenta", width=7),
            marker=dict(size=3), name="Pháp tuyến"
        )
        fig.add_scatter3d(
            x=[gx0], y=[gy0], z=[z0], mode="markers+text",
            marker=dict(size=7, color="black"),
            text=["M₀"], textposition="top center", name="M₀"
        )
        fig.update_layout(
            height=680, margin=dict(l=0, r=0, t=25, b=0),
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.8)
            ),
            legend=dict(orientation="h", y=-0.08)
        )
        with right:
            st.plotly_chart(fig, use_container_width=True)

# ================================================================
# TAB 3: BIẾN THIÊN
with tab_change:
    st.subheader("Đường mức và đạo hàm theo hướng")
    st.caption("Gradient chỉ phụ thuộc vào hàm số và điểm đang xét. Khi đổi hướng u, v, gradient không đổi; đạo hàm theo hướng sẽ thay đổi.")
    controls, plot_col = st.columns([1, 1.8], gap="large")
    with controls:
        st.markdown("**Điểm A**")
        xa = number_control("xA", "var_xa", 1.0)
        ya = number_control("yA", "var_ya", 1.0)
        st.markdown("**Điểm B**")
        xb = number_control("xB", "var_xb", -1.5)
        yb = number_control("yB", "var_yb", -0.5)
        st.markdown("**Chọn hướng tại A**")
        theta_u = st.slider("Góc θᵤ (độ)", 0, 360, 45)
        theta_v = st.slider("Góc θᵥ (độ)", 0, 360, 135)

    VA = point_values(functions, xa, ya)
    VB = point_values(functions, xb, yb)
    u = np.array([np.cos(np.radians(theta_u)),
                  np.sin(np.radians(theta_u))])
    v = np.array([np.cos(np.radians(theta_v)),
                  np.sin(np.radians(theta_v))])
    gradA = np.array([VA["fx"], VA["fy"]])
    normA = np.linalg.norm(gradA)
    Du = float(np.dot(gradA, u))
    Dv = float(np.dot(gradA, v))

    with plot_col:
        grid = np.linspace(-LIMIT, LIMIT, 220)
        X, Y = np.meshgrid(grid, grid)
        Z = safe_grid(functions["f"], X, Y)
        fig, ax = plt.subplots(figsize=(8, 7))
        cf = ax.contourf(X, Y, Z, levels=24, cmap="viridis", alpha=0.88)
        contours = ax.contour(X, Y, Z, levels=24, colors="black",
                              linewidths=0.45, alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f")
        q = np.linspace(-LIMIT, LIMIT, 17)
        XQ, YQ = np.meshgrid(q, q)
        Gx = safe_grid(functions["fx"], XQ, YQ)
        Gy = safe_grid(functions["fy"], XQ, YQ)
        mag = np.hypot(Gx, Gy)
        mag[mag == 0] = np.nan
        ax.quiver(XQ, YQ, Gx/mag, Gy/mag, color="white",
                  alpha=0.55, scale=27, width=0.003)

        for px, py, point, color, label in [
            (xa, ya, VA, "red", "A"),
            (xb, yb, VB, "deepskyblue", "B")
        ]:
            ax.scatter(px, py, s=90, color=color, edgecolor="black",
                       linewidth=0.8, zorder=6)
            ax.annotate(label, (px, py), xytext=(8, 8),
                        textcoords="offset points", color="white",
                        weight="bold", bbox=dict(boxstyle="round",
                        fc="black", alpha=0.55))
            # Gradient riêng tại từng điểm
            grad = np.array([point["fx"], point["fy"]])
            length = np.linalg.norm(grad)
            if length > 1e-9:
                direction = grad / length
                ax.annotate("", xy=(px+direction[0]*0.75,
                                    py+direction[1]*0.75),
                            xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>",
                                            color="yellow", lw=2.8))

        # u và v cùng gốc tại A; thay đổi góc sẽ làm hai mũi tên này quay
        ax.annotate("", xy=(xa+u[0], ya+u[1]), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color="magenta", lw=2.8))
        ax.text(xa+u[0]*1.08, ya+u[1]*1.08, "u", color="magenta",
                weight="bold", fontsize=12)
        ax.annotate("", xy=(xa+v[0], ya+v[1]), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color="orange", lw=2.8))
        ax.text(xa+v[0]*1.08, ya+v[1]*1.08, "v", color="orange",
                weight="bold", fontsize=12)

        ax.set_xlim(-LIMIT, LIMIT)
        ax.set_ylim(-LIMIT, LIMIT)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Bản đồ đường mức và các hướng tại A")
        ax.grid(alpha=0.12)
        fig.colorbar(cf, ax=ax, label="Giá trị f(x,y)", shrink=0.82)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Kết quả hướng và gradient
    st.markdown("#### Kết quả tại A")
    k1, k2, k3 = st.columns(3)
    k1.metric("|∇f(A)|", f"{normA:.4f}")
    k2.metric("Dᵤf(A)", f"{Du:.4f}",
              delta="Tăng" if Du > 1e-8 else ("Giảm" if Du < -1e-8 else "Không đổi"))
    k3.metric("Dᵥf(A)", f"{Dv:.4f}",
              delta="Tăng" if Dv > 1e-8 else ("Giảm" if Dv < -1e-8 else "Không đổi"))
    st.latex(
        rf"\nabla f(A)=({VA['fx']:.4f},{VA['fy']:.4f}),\quad "
        rf"\vec u=({u[0]:.3f},{u[1]:.3f}),\quad "
        rf"D_{{\vec u}}f(A)=\nabla f(A)\cdot\vec u={Du:.4f}"
    )
    st.latex(
        rf"\vec v=({v[0]:.3f},{v[1]:.3f}),\quad "
        rf"D_{{\vec v}}f(A)=\nabla f(A)\cdot\vec v={Dv:.4f}"
    )
    st.caption("D > 0: hàm tăng theo hướng đã chọn. D < 0: hàm giảm. D = 0: biến thiên tức thời bằng 0 theo hướng đó.")

    st.markdown("#### So sánh A và B")
    def point_row(label, W):
        return {
            "Điểm": label,
            "f": round(W["f"], 4),
            "fₓ": round(W["fx"], 4),
            "fᵧ": round(W["fy"], 4),
            "|∇f|": round(float(np.hypot(W["fx"], W["fy"])), 4),
            "fₓₓ": round(W["fxx"], 4),
            "fₓᵧ": round(W["fxy"], 4),
            "fᵧᵧ": round(W["fyy"], 4),
        }
    st.dataframe(pd.DataFrame([
        point_row("A", VA), point_row("B", VB)
    ]), hide_index=True, use_container_width=True)
    normB = float(np.hypot(VB["fx"], VB["fy"]))
    st.write(
        f"- Tại A: |∇f(A)| = **{normA:.4f}**.\n"
        f"- Tại B: |∇f(B)| = **{normB:.4f}**.\n"
        f"- "
        + ("Hàm biến thiên nhanh hơn tại A." if normA > normB + 1e-8
           else "Hàm biến thiên nhanh hơn tại B." if normB > normA + 1e-8
           else "Độ lớn gradient tại A và B bằng nhau.")
    )

    st.markdown("#### Điểm tới hạn")
    st.caption("Tìm gần đúng các điểm thỏa fₓ = 0 và fᵧ = 0 trong miền khảo sát [-3, 3] × [-3, 3].")
    if st.button("Tìm điểm tới hạn"):
        critical = []
        for sx in np.linspace(-LIMIT, LIMIT, 7):
            for sy in np.linspace(-LIMIT, LIMIT, 7):
                try:
                    sol = sp.nsolve((derivatives["fx"], derivatives["fy"]),
                                    (x, y), (sx, sy), tol=1e-10,
                                    maxsteps=50, verify=False)
                    px, py = float(sol[0]), float(sol[1])
                    if abs(px) > LIMIT or abs(py) > LIMIT:
                        continue
                    vv = point_values(functions, px, py)
                    if max(abs(vv["fx"]), abs(vv["fy"])) > 1e-5:
                        continue
                    if any(np.hypot(px-qx, py-qy) < 1e-3
                           for qx, qy, _ in critical):
                        continue
                    determinant = vv["fxx"]*vv["fyy"] - vv["fxy"]**2
                    if determinant > 1e-8:
                        kind = "Cực tiểu" if vv["fxx"] > 0 else (
                            "Cực đại" if vv["fxx"] < 0 else "Chưa kết luận")
                    elif determinant < -1e-8:
                        kind = "Điểm yên ngựa"
                    else:
                        kind = "Chưa kết luận (D = 0)"
                    critical.append((px, py, kind))
                except Exception:
                    pass
        if critical:
            st.dataframe(pd.DataFrame([
                {"x": round(px, 5), "y": round(py, 5), "Phân loại": kind}
                for px, py, kind in critical
            ]), hide_index=True, use_container_width=True)
        else:
            st.info("Không tìm thấy điểm tới hạn trong miền khảo sát bằng phương pháp số.")

    with st.expander("Cách đọc bản đồ đường mức"):
        st.markdown(
            "- Gradient ∇f vuông góc với đường mức và hướng về phía hàm tăng nhanh nhất.\n"
            "- −∇f hướng về phía hàm giảm nhanh nhất.\n"
            "- Đường mức càng sít nhau thì thường |∇f| càng lớn.\n"
            "- Nếu hướng u tiếp tuyến với đường mức tại A thì Dᵤf(A) = 0.\n"
            "- Dᵤf(A) > 0: hàm tăng theo u; Dᵤf(A) < 0: hàm giảm theo u."
        )
# ============================================================
# TAB 4: ỨNG DỤNG CỦA ĐẠO HÀM RIÊNG
# Dán sau toàn bộ nội dung TAB 3
# ============================================================

with tab_app:

    # --------------------------------------------------------
    # CSS riêng cho tab ứng dụng
    # --------------------------------------------------------
    st.markdown("""
    <style>
    .app-hero {
        background: linear-gradient(120deg, #142D4E, #245B78);
        padding: 30px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
    }
    .app-hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .app-hero p {
        color: #E4EDF5;
        font-size: 15px;
        margin-bottom: 0;
    }
    .app-kicker {
        color: #A9D8EF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .app-section {
        padding: 15px 0 7px 0;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 16px;
    }
    .app-section h3 {
        margin-bottom: 5px;
    }
    .app-lead {
        font-size: 16px;
        line-height: 1.7;
    }
    .app-note {
        background: rgba(80,130,160,0.10);
        border-left: 4px solid #3985A8;
        padding: 12px 16px;
        border-radius: 5px;
        margin: 12px 0;
        line-height: 1.65;
    }
    .app-paper {
        background: rgba(130,130,130,0.06);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }
    .app-small {
        color: #777;
        font-size: 12px;
    }
    .app-source {
        font-size: 13px;
        margin-top: 8px;
    }
    .case-banner {
        background: linear-gradient(120deg, #493A2D, #826442);
        padding: 25px;
        border-radius: 14px;
        color: white;
        margin: 30px 0 20px 0;
    }
    .case-banner h2 {
        color: white;
        margin-bottom: 8px;
    }
    .case-banner p {
        color: #F5EBDD;
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------
    st.markdown("""
    <div class="app-hero">
        <div class="app-kicker">Mathematics in real life</div>
        <h1>ĐẠO HÀM RIÊNG TRONG THỰC TIỄN</h1>
        <p>
            Từ việc phân tích địa hình, nghiên cứu sinh trưởng vi sinh vật,
            tối ưu hóa mô hình trí tuệ nhân tạo đến dự báo thời tiết.
            Đạo hàm riêng giúp chúng ta hiểu một đại lượng thay đổi như thế nào
            khi một yếu tố đầu vào thay đổi và các yếu tố còn lại được giữ cố định.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Vì sao đạo hàm riêng có nhiều ứng dụng?

    Trong thực tế, một hiện tượng thường chịu ảnh hưởng đồng thời của nhiều yếu tố.
    Ví dụ, sản lượng phụ thuộc vào vốn và lao động; cảm giác nóng phụ thuộc vào
    nhiệt độ và độ ẩm; tốc độ sinh trưởng phụ thuộc vào điều kiện môi trường.

    Nếu biểu diễn một đại lượng bằng hàm nhiều biến:

    """)
    st.latex(r"z=f(x,y,\ldots)")

    st.markdown("""
    thì đạo hàm riêng cho phép khảo sát ảnh hưởng của **từng biến riêng biệt**,
    trong khi tạm thời giữ các biến còn lại không đổi.

    Đây là cơ sở để phân tích độ nhạy, dự đoán sự thay đổi và hỗ trợ ra quyết định.
    """)

    # ========================================================
    # PHẦN I: CÁC ỨNG DỤNG
    # ========================================================
    st.markdown("---")
    st.markdown("## PHẦN I · CÁC LĨNH VỰC ỨNG DỤNG")
    st.caption(
        "Mỗi ví dụ dưới đây đều đi từ bài toán thực tế đến mô hình toán học "
        "và cách sử dụng đạo hàm riêng."
    )

    # --------------------------------------------------------
    # 1. KIẾN TRÚC - QUY HOẠCH
    # --------------------------------------------------------
    with st.expander(
        "01 · KIẾN TRÚC VÀ QUY HOẠCH — Phân tích địa hình và thoát nước",
        expanded=True
    ):
        st.markdown("### Bài toán thực tế")

        col_img, col_text = st.columns([1, 1.5])
        with col_img:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/"
                "Contour_lines.svg/640px-Contour_lines.svg.png",
                caption="Minh họa bản đồ đường đồng mức",
                use_container_width=True
            )
        with col_text:
            st.markdown("""
            Khi quy hoạch khu dân cư, công viên hoặc xây dựng công trình,
            kỹ sư cần biết địa hình cao thấp ra sao, dốc theo hướng nào và
            nước mưa có xu hướng chảy về đâu.

            Bản đồ đường đồng mức cho biết các vị trí có cùng độ cao.
            Tuy nhiên, để định lượng độ dốc tại từng điểm, người ta cần
            phân tích sự thay đổi của độ cao theo các hướng khác nhau.
            """)

        st.markdown("### Mô hình toán học")
        st.latex(r"z=f(x,y)")

        st.markdown("""
        - \(x,y\): tọa độ ngang của một điểm trên bản đồ.
        - \(z\): độ cao địa hình tại vị trí \((x,y)\).
        """)

        st.markdown("### Công thức đạo hàm riêng")
        st.latex(
            r"f_x=\frac{\partial z}{\partial x},\qquad"
            r"f_y=\frac{\partial z}{\partial y}"
        )
        st.markdown("""
        - \(f_x\): tốc độ thay đổi độ cao theo hướng trục \(x\).
        - \(f_y\): tốc độ thay đổi độ cao theo hướng trục \(y\).

        Từ hai thành phần này, ta xác định gradient:
        """)
        st.latex(
            r"\nabla f=(f_x,f_y),\qquad"
            r"\|\nabla f\|=\sqrt{f_x^2+f_y^2}"
        )
        st.markdown("""
        Độ lớn gradient thể hiện mức độ dốc lớn nhất tại điểm đang xét.
        Nếu cần biểu diễn góc nghiêng của địa hình so với mặt phẳng ngang:
        """)
        st.latex(r"\theta=\arctan(\|\nabla f\|)")

        st.markdown("### Người ta sử dụng kết quả để làm gì?")
        st.markdown("""
        1. **Lập bản đồ độ dốc:** xác định vùng đất bằng phẳng, dốc vừa hoặc dốc lớn.
        2. **Thiết kế thoát nước:** xác định hướng giảm độ cao để dự kiến hướng dòng chảy.
        3. **Quy hoạch xây dựng:** phát hiện khu vực cần san lấp, gia cố hoặc hạn chế xây dựng.
        4. **Thiết kế đường giao thông:** đánh giá độ dốc của tuyến đường để đảm bảo an toàn.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Phân tích:</b> Nếu độ lớn gradient lớn, địa hình thay đổi độ cao nhanh
        trong một khoảng cách ngắn. Điều này có thể làm tăng yêu cầu về san lấp,
        gia cố mái dốc và thiết kế thoát nước. Đạo hàm riêng vì thế là cơ sở toán học
        cho các phép phân tích địa hình trong GIS.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "**Mức độ ứng dụng:** Được sử dụng rộng rãi trong GIS và phân tích "
            "địa hình số. Các phần mềm có thể tính độ dốc từ mô hình độ cao số (DEM)."
        )
        st.markdown(
            "[Nguồn: ArcGIS – How Slope Works]"
            "(https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm)  \n"
            "[Nguồn: Google Earth Engine – Terrain Slope]"
            "(https://developers.google.com/earth-engine/apidocs/ee-terrain-slope)"
        )

    # --------------------------------------------------------
    # 2. VI SINH VẬT HỌC
    # --------------------------------------------------------
    with st.expander(
        "02 · VI SINH VẬT HỌC — Phân tích tốc độ sinh trưởng",
        expanded=False
    ):
        st.markdown("### Bài toán thực tế")
        st.markdown("""
        Trong sản xuất thực phẩm, dược phẩm và công nghệ sinh học,
        người ta thường nuôi cấy vi sinh vật trong môi trường có kiểm soát.

        Tốc độ sinh trưởng có thể chịu ảnh hưởng bởi nhiệt độ, nồng độ chất dinh dưỡng,
        độ pH và nhiều yếu tố khác. Người nghiên cứu cần biết yếu tố nào đang
        ảnh hưởng mạnh đến tốc độ sinh trưởng để điều chỉnh môi trường nuôi cấy.
        """)

        st.markdown("### Mô hình tổng quát")
        st.latex(r"R=f(T,C)")
        st.markdown("""
        - \(R\): tốc độ sinh trưởng của vi sinh vật.
        - \(T\): nhiệt độ môi trường.
        - \(C\): nồng độ cơ chất – nguồn dinh dưỡng mà vi sinh vật sử dụng.
        """)

        st.markdown("### Vai trò của đạo hàm riêng")
        st.latex(
            r"\frac{\partial R}{\partial T}"
            r"\quad\text{và}\quad"
            r"\frac{\partial R}{\partial C}"
        )
        st.markdown("""
        - \(\partial R/\partial T\): mức thay đổi tốc độ sinh trưởng khi nhiệt độ
        thay đổi, còn nồng độ cơ chất được giữ cố định.
        - \(\partial R/\partial C\): mức thay đổi tốc độ sinh trưởng khi nồng độ
        cơ chất thay đổi, còn nhiệt độ được giữ cố định.
        """)

        st.markdown("### Ví dụ cụ thể: mô hình Monod")
        st.markdown("""
        Mô hình Monod mô tả mối quan hệ giữa nồng độ cơ chất và tốc độ sinh trưởng
        riêng của vi sinh vật:
        """)
        st.latex(
            r"\mu(C)=\mu_{\max}\frac{C}{K_s+C}"
        )
        st.markdown("""
        Trong đó:
        - \(\mu\): tốc độ sinh trưởng riêng.
        - \(\mu_{\max}\): tốc độ sinh trưởng riêng cực đại.
        - \(C\): nồng độ cơ chất.
        - \(K_s\): hằng số bán bão hòa.
        """)

        st.markdown("Đạo hàm theo nồng độ cơ chất:")
        st.latex(
            r"\frac{d\mu}{dC}"
            r"=\frac{\mu_{\max}K_s}{(K_s+C)^2}"
        )

        st.markdown("""
        **Ý nghĩa:** Khi nồng độ cơ chất tăng, tốc độ sinh trưởng có xu hướng tăng.
        Tuy nhiên, đạo hàm giảm dần khi \(C\) tăng, cho thấy lượng cơ chất bổ sung
        thêm đem lại mức tăng trưởng ngày càng nhỏ.

        Người nghiên cứu có thể sử dụng mô hình để phân tích mức độ đáp ứng của
        vi sinh vật với nguồn dinh dưỡng và hỗ trợ lựa chọn điều kiện nuôi cấy.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Lưu ý khoa học:</b> Phương trình Monod cơ bản chỉ mô tả ảnh hưởng của
        nồng độ cơ chất. Nó không trực tiếp mô tả ảnh hưởng của nhiệt độ.
        Muốn mô hình hóa đồng thời nhiệt độ và dinh dưỡng cần xây dựng mô hình mở rộng.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "**Mức độ ứng dụng:** Mô hình Monod là mô hình kinh điển trong động học "
            "sinh trưởng vi sinh vật và công nghệ sinh học."
        )
        st.markdown(
            "[Nguồn: Monod – The Growth of Bacterial Cultures]"
            "(https://doi.org/10.1146/annurev.mi.03.100149.002103)"
        )

    # --------------------------------------------------------
    # 3. KINH TẾ HỌC
    # --------------------------------------------------------
    with st.expander(
        "03 · KINH TẾ HỌC — Năng suất cận biên của vốn và lao động",
        expanded=False
    ):
        st.markdown("### Bài toán thực tế")
        st.markdown("""
        Một doanh nghiệp muốn tăng sản lượng có thể lựa chọn đầu tư thêm máy móc,
        thiết bị hoặc thuê thêm lao động. Tuy nhiên, doanh nghiệp cần ước lượng
        việc tăng một yếu tố sẽ làm sản lượng thay đổi bao nhiêu khi các yếu tố
        còn lại không đổi.
        """)

        st.markdown("### Hàm sản xuất Cobb–Douglas")
        st.latex(r"Y=A K^\alpha L^\beta")

        st.markdown("""
        Trong đó:
        - \(Y\): tổng sản lượng.
        - \(K\): vốn, máy móc và thiết bị.
        - \(L\): lượng lao động.
        - \(A\): hệ số phản ánh công nghệ và hiệu quả sản xuất.
        - \(\alpha,\beta\): hệ số phản ánh mức độ đóng góp của vốn và lao động.
        """)

        st.markdown("### Đạo hàm riêng – Sản phẩm cận biên")
        st.latex(
            r"MP_K=\frac{\partial Y}{\partial K}"
            r"=A\alpha K^{\alpha-1}L^\beta"
        )
        st.latex(
            r"MP_L=\frac{\partial Y}{\partial L}"
            r"=A\beta K^\alpha L^{\beta-1}"
        )

        st.markdown("""
        - \(MP_K\): sản lượng tăng thêm khi vốn tăng một đơn vị, giữ lao động không đổi.
        - \(MP_L\): sản lượng tăng thêm khi lao động tăng một đơn vị, giữ vốn không đổi.
        """)

        st.markdown("### Từ đạo hàm đến quyết định kinh doanh")
        st.latex(
            r"\Delta Y\approx"
            r"\frac{\partial Y}{\partial K}\Delta K"
            r"+\frac{\partial Y}{\partial L}\Delta L"
        )
        st.markdown("""
        Công thức trên cho phép ước lượng gần đúng mức thay đổi sản lượng khi doanh
        nghiệp điều chỉnh vốn và lao động một lượng nhỏ.

        Ví dụ, nếu doanh nghiệp chỉ tăng vốn thì:
        """)
        st.latex(
            r"\Delta Y\approx"
            r"\frac{\partial Y}{\partial K}\Delta K"
        )

        st.markdown("""
        Doanh nghiệp có thể dùng thông tin này để:
        1. Ước lượng lợi ích của việc đầu tư thêm máy móc.
        2. So sánh hiệu quả tăng vốn với tăng lao động.
        3. Phân tích năng suất cận biên giảm dần.
        4. Hỗ trợ phân bổ nguồn lực sản xuất.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Phân tích:</b> Đạo hàm riêng cung cấp mức sản lượng tăng thêm ở biên,
        chứ không tự quyết định doanh nghiệp nên đầu tư bao nhiêu.
        Để đưa ra quyết định, doanh nghiệp còn phải so sánh lợi ích cận biên
        với chi phí vốn, tiền lương và giá bán sản phẩm.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "[Nguồn tham khảo: Production Theory – Econlib]"
            "(https://www.econlib.org/library/Enc/ProductionTheory.html)"
        )

    # --------------------------------------------------------
    # 4. AI / MACHINE LEARNING
    # --------------------------------------------------------
    with st.expander(
        "04 · TRÍ TUỆ NHÂN TẠO — Tối ưu mô hình bằng Gradient Descent",
        expanded=False
    ):
        st.markdown("### Bài toán thực tế")
        st.markdown("""
        Khi huấn luyện một mô hình Machine Learning, mô hình tạo ra dự đoán
        và so sánh với kết quả thực tế. Mức độ sai lệch được đo bằng hàm mất mát
        (Loss Function).

        Mục tiêu của quá trình huấn luyện là điều chỉnh các tham số của mô hình
        sao cho hàm mất mát giảm xuống.
        """)

        st.markdown("### Hàm mất mát nhiều biến")
        st.latex(r"L=f(w_1,w_2,\ldots,w_n)")

        st.markdown("""
        - \(L\): giá trị hàm mất mát.
        - \(w_1,w_2,\ldots,w_n\): các trọng số của mô hình.
        """)

        st.markdown("### Đạo hàm riêng của hàm mất mát")
        st.latex(
            r"\frac{\partial L}{\partial w_i}"
        )
        st.markdown("""
        Đạo hàm này cho biết hàm mất mát thay đổi như thế nào khi trọng số
        \(w_i\) thay đổi một lượng nhỏ, trong khi các trọng số còn lại được giữ cố định.
        """)

        st.markdown("Gradient của hàm mất mát:")
        st.latex(
            r"\nabla L="
            r"\left("
            r"\frac{\partial L}{\partial w_1},"
            r"\frac{\partial L}{\partial w_2},"
            r"\ldots,"
            r"\frac{\partial L}{\partial w_n}"
            r"\right)"
        )

        st.markdown("### Quy tắc cập nhật trọng số")
        st.latex(
            r"w_i^{(t+1)}="
            r"w_i^{(t)}-\eta\frac{\partial L}{\partial w_i}"
        )

        st.markdown("""
        Trong đó:
        - \(w_i^{(t)}\): trọng số ở bước hiện tại.
        - \(w_i^{(t+1)}\): trọng số sau khi cập nhật.
        - \(\eta\): learning rate – tốc độ học.
        - \(\partial L/\partial w_i\): đạo hàm riêng của hàm mất mát theo trọng số.
        """)

        st.markdown("""
        **Người ta tính được gì?**

        - Hướng điều chỉnh các trọng số để làm giảm hàm mất mát.
        - Mức độ ảnh hưởng của từng trọng số đến sai số dự đoán.
        - Giá trị trọng số mới sau mỗi lần cập nhật.

        Trong mạng nơ-ron, thuật toán Backpropagation sử dụng quy tắc dây chuyền
        để tính gradient của hàm mất mát theo các trọng số.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Phân tích:</b> Nếu đạo hàm theo một trọng số có giá trị dương,
        tăng trọng số đó một lượng nhỏ có xu hướng làm tăng loss.
        Gradient Descent cập nhật theo hướng ngược gradient để tìm cách giảm loss.
        Learning rate quá lớn có thể khiến quá trình cập nhật không ổn định;
        quá nhỏ có thể khiến mô hình học rất chậm.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "**Mức độ ứng dụng:** Gradient và đạo hàm riêng là thành phần cốt lõi "
            "trong nhiều thuật toán huấn luyện Machine Learning và Deep Learning."
        )
        st.markdown(
            "[Nguồn: Deep Learning – Optimization for Training Deep Models]"
            "(https://www.deeplearningbook.org/contents/optimization.html)"
        )

    # --------------------------------------------------------
    # 5. XỬ LÝ ẢNH
    # --------------------------------------------------------
    with st.expander(
        "05 · XỬ LÝ ẢNH SỐ — Phát hiện đường biên của vật thể",
        expanded=False
    ):
        st.markdown("### Bài toán thực tế")
        st.markdown("""
        Một ảnh số có thể được biểu diễn thành ma trận các pixel.
        Mỗi pixel mang thông tin về cường độ sáng hoặc màu sắc.

        Để máy tính nhận biết đường viền của một vật thể, cần tìm những vị trí
        mà cường độ sáng thay đổi mạnh giữa các pixel lân cận.
        """)

        st.markdown("### Mô hình toán học")
        st.latex(r"I=I(x,y)")

        st.markdown("""
        - \(I(x,y)\): cường độ sáng tại pixel có tọa độ \((x,y)\).
        - \(x,y\): tọa độ của pixel trong ảnh.
        """)

        st.markdown("### Đạo hàm riêng theo hai hướng")
        st.latex(
            r"I_x=\frac{\partial I}{\partial x},\qquad"
            r"I_y=\frac{\partial I}{\partial y}"
        )
        st.markdown("""
        - \(I_x\): mức thay đổi cường độ sáng theo chiều ngang.
        - \(I_y\): mức thay đổi cường độ sáng theo chiều dọc.
        """)

        st.markdown("Độ lớn gradient ảnh:")
        st.latex(
            r"G=\sqrt{I_x^2+I_y^2}"
        )

        st.markdown("""
        **Ý nghĩa:** Nếu \(G\) lớn, cường độ sáng thay đổi mạnh tại vị trí đó.
        Đây thường là dấu hiệu của vùng biên giữa hai vùng ảnh có độ sáng khác nhau.
        """)

        st.markdown("### Người ta sử dụng kết quả để làm gì?")
        st.markdown("""
        1. Phát hiện đường biên của vật thể.
        2. Hỗ trợ phân đoạn ảnh.
        3. Trích xuất đặc trưng hình dạng.
        4. Làm bước tiền xử lý cho các hệ thống thị giác máy tính.
        """)

        st.markdown("### Liên hệ với toán tử Sobel")
        st.markdown("""
        Trong ảnh số, đạo hàm thường được xấp xỉ bằng sai phân hữu hạn.
        Toán tử Sobel sử dụng hai kernel tích chập để xấp xỉ đạo hàm theo
        phương ngang và phương dọc. Sau đó, độ lớn gradient được dùng để
        đánh giá mức độ thay đổi cường độ sáng.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Phân tích:</b> Đạo hàm riêng không trực tiếp nhận diện tên vật thể.
        Nó giúp phát hiện các vùng có thay đổi cường độ sáng; các bước xử lý
        tiếp theo mới kết hợp đường biên và đặc trưng để phân đoạn hoặc nhận dạng.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "[Nguồn tham khảo: Springer – Image Processing]"
            "(https://link.springer.com/chapter/10.1007/978-3-030-01364-6_3)"
        )

    # --------------------------------------------------------
    # 6. KHÍ TƯỢNG HỌC
    # --------------------------------------------------------
    with st.expander(
        "06 · KHÍ TƯỢNG HỌC — Phân tích gradient áp suất khí quyển",
        expanded=False
    ):
        st.markdown("### Bài toán thực tế")
        st.markdown("""
        Áp suất khí quyển không giống nhau ở mọi vị trí.
        Sự chênh lệch áp suất giữa các khu vực có liên quan đến chuyển động
        của không khí và sự hình thành gió.

        Các nhà khí tượng cần phân tích áp suất thay đổi theo không gian
        như thế nào để xây dựng mô hình dự báo thời tiết.
        """)

        st.markdown("### Mô hình toán học")
        st.latex(r"P=P(x,y,z,t)")

        st.markdown("""
        - \(P\): áp suất khí quyển.
        - \(x,y,z\): tọa độ không gian.
        - \(t\): thời gian.
        """)

        st.markdown("### Đạo hàm riêng theo không gian")
        st.latex(
            r"\frac{\partial P}{\partial x},\qquad"
            r"\frac{\partial P}{\partial y},\qquad"
            r"\frac{\partial P}{\partial z}"
        )

        st.markdown("Gradient áp suất:")
        st.latex(
            r"\nabla P="
            r"\left("
            r"\frac{\partial P}{\partial x},"
            r"\frac{\partial P}{\partial y},"
            r"\frac{\partial P}{\partial z}"
            r"\right)"
        )

        st.markdown("""
        **Người ta tính được gì?**

        - Hướng áp suất tăng nhanh nhất.
        - Mức độ thay đổi áp suất giữa các khu vực.
        - Thành phần gradient áp suất để đưa vào các phương trình mô tả
        chuyển động của khí quyển.
        """)

        st.markdown("""
        <div class="app-note">
        <b>Phân tích:</b> Gradient áp suất có liên quan đến lực gradient áp suất,
        một trong các yếu tố làm không khí chuyển động. Tuy nhiên, hướng gió thực tế
        còn chịu tác động của lực Coriolis, ma sát và nhiều yếu tố khí động lực học khác.
        Vì vậy, không thể chỉ dựa vào gradient áp suất để dự báo chính xác hướng gió
        hoặc đường đi của bão.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "[Nguồn: National Weather Service – Atmospheric Pressure]"
            "(https://www.weather.gov/jetstream/pressure)"
        )

    # ========================================================
    # PHẦN II: NGHIÊN CỨU CHÍNH - HEAT INDEX
    # ========================================================
    st.markdown("---")

    st.markdown("""
    <div class="case-banner">
        <div class="app-kicker">Main case study · An Giang</div>
        <h2>PHÂN TÍCH CHỈ SỐ NHIỆT HEAT INDEX</h2>
        <p>
            Ứng dụng đạo hàm riêng để nghiên cứu ảnh hưởng của nhiệt độ
            và độ ẩm đến cảm giác nóng của con người.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 2.1. Đặt vấn đề

    Trong những ngày nắng nóng, nhiệt độ không khí không phải yếu tố duy nhất
    quyết định mức độ nóng mà con người cảm nhận. Độ ẩm tương đối cũng có ảnh hưởng
    đáng kể đến khả năng tản nhiệt của cơ thể.

    Khi độ ẩm cao, mồ hôi khó bay hơi hơn, khiến cơ thể khó giải phóng nhiệt.
    Vì vậy, cùng một nhiệt độ không khí nhưng ở những mức độ ẩm khác nhau,
    con người có thể cảm nhận mức nóng khác nhau.

    **Heat Index (chỉ số nhiệt)** là đại lượng kết hợp nhiệt độ không khí
    và độ ẩm tương đối để ước lượng mức nóng mà cơ thể cảm nhận.
    """)

    st.markdown("""
    <div class="app-note">
    <b>Câu hỏi nghiên cứu:</b><br>
    1. Khi nhiệt độ tăng, Heat Index thay đổi như thế nào nếu độ ẩm không đổi?<br>
    2. Khi độ ẩm tăng, Heat Index thay đổi như thế nào nếu nhiệt độ không đổi?<br>
    3. Tại các trạm khí tượng ở An Giang, yếu tố nào có ảnh hưởng lớn hơn
    đến mức thay đổi của Heat Index tại thời điểm quan sát?
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 2.2. MÔ HÌNH ROTHFUSZ
    # --------------------------------------------------------
    st.markdown("### 2.2. Mô hình toán học – Công thức Rothfusz")

    st.markdown("""
    Chỉ số nhiệt được mô hình hóa như một hàm của hai biến đầu vào:
    nhiệt độ không khí và độ ẩm tương đối.
    """)
    st.latex(r"HI=f(T,RH)")

    st.markdown("""
    Trong nghiên cứu này sử dụng phương trình hồi quy Rothfusz,
    được National Weather Service sử dụng trong tính toán Heat Index.
    Công thức gốc sử dụng nhiệt độ Fahrenheit và độ ẩm tương đối theo phần trăm.
    """)

    with st.expander("Xem đầy đủ công thức Rothfusz và giải thích hệ số", expanded=True):
        st.latex(
            r"\begin{aligned}"
            r"HI={}&-42.379+2.04901523T+10.14333127RH\\"
            r"&-0.22475541T(RH)-6.83783\times10^{-3}T^2\\"
            r"&-5.481717\times10^{-2}RH^2"
            r"+1.22874\times10^{-3}T^2(RH)\\"
            r"&+8.5282\times10^{-4}T(RH)^2"
            r"-1.99\times10^{-6}T^2(RH)^2"
            r"\end{aligned}"
        )

        st.markdown("""
        **Các biến trong công thức:**

        | Ký hiệu | Ý nghĩa | Đơn vị |
        |---|---|---|
        | \(T\) | Nhiệt độ không khí | °F |
        | \(RH\) | Độ ẩm tương đối | % |
        | \(HI\) | Chỉ số nhiệt | °F |

        Các hệ số là hệ số hồi quy thực nghiệm được xác định trong mô hình
        Rothfusz. Chúng không phải các đại lượng đo trực tiếp tại trạm khí tượng.

        **Chuyển đổi đơn vị:**

        \[
        T_F=\frac{9}{5}T_C+32
        \]

        \[
        HI_C=\frac{5}{9}(HI_F-32)
        \]

        Trong đó \(T_C\) và \(HI_C\) lần lượt là nhiệt độ và chỉ số nhiệt
        biểu diễn theo độ Celsius.
        """)

        st.markdown(
            "[Nguồn chính thức: NOAA/National Weather Service – "
            "Heat Index Equation]"
            "(https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml)"
        )

    # --------------------------------------------------------
    # 2.3. DỮ LIỆU THỰC TẾ
    # --------------------------------------------------------
    st.markdown("### 2.3. Dữ liệu khí tượng tại An Giang")

    st.markdown("""
    Bảng dưới đây tổng hợp dữ liệu tại bốn trạm khí tượng được sử dụng trong
    nghiên cứu. Nhiệt độ và độ ẩm là hai biến đầu vào của mô hình;
    Heat Index và các đạo hàm riêng là kết quả tính toán được cung cấp
    trong bộ số liệu nghiên cứu.
    """)

    weather_data = pd.DataFrame({
        "Trạm khí tượng": [
            "Phú Quốc",
            "Rạch Giá",
            "Thổ Chu",
            "Châu Đốc"
        ],
        "Nhiệt độ (°C)": [31.4, 33.2, 34.2, 36.1],
        "Độ ẩm (%)": [69, 54, 61, 53],
        "Heat Index (°C)": [38.4, 37.9, 43.2, 44.8],
        "∂HI/∂T": [2.72, 2.26, 2.87, 2.75],
        "∂HI/∂RH": [0.32, 0.33, 0.44, 0.49]
    })

    st.dataframe(
        weather_data.style.format({
            "Nhiệt độ (°C)": "{:.1f}",
            "Độ ẩm (%)": "{:.0f}",
            "Heat Index (°C)": "{:.1f}",
            "∂HI/∂T": "{:.2f}",
            "∂HI/∂RH": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Lưu ý: Các giá trị đạo hàm trong bảng được giữ theo bộ kết quả nghiên cứu "
        "đã cung cấp. Cần đối chiếu lại phép tính, quy ước đơn vị và thời điểm "
        "quan trắc với nguồn dữ liệu gốc trước khi sử dụng như kết quả chính thức."
    )

    # --------------------------------------------------------
    # 2.4. ĐẠO HÀM RIÊNG
    # --------------------------------------------------------
    st.markdown("### 2.4. Tính đạo hàm riêng và ý nghĩa")

    st.markdown("""
    Vì Heat Index phụ thuộc đồng thời vào nhiệt độ và độ ẩm, ta sử dụng
    hai đạo hàm riêng để phân tích ảnh hưởng của từng yếu tố.
    """)

    col_t, col_rh = st.columns(2)

    with col_t:
        st.markdown("**Đạo hàm theo nhiệt độ**")
        st.latex(r"\frac{\partial HI}{\partial T}")
        st.markdown("""
        Giữ độ ẩm không đổi, đạo hàm cho biết Heat Index thay đổi gần đúng
        bao nhiêu khi nhiệt độ tăng thêm một đơn vị.
        """)

    with col_rh:
        st.markdown("**Đạo hàm theo độ ẩm**")
        st.latex(r"\frac{\partial HI}{\partial RH}")
        st.markdown("""
        Giữ nhiệt độ không đổi, đạo hàm cho biết Heat Index thay đổi gần đúng
        bao nhiêu khi độ ẩm tăng thêm một điểm phần trăm.
        """)

    st.markdown("#### Công thức đạo hàm của mô hình Rothfusz")

    st.markdown("Đạo hàm theo nhiệt độ Fahrenheit:")
    st.latex(
        r"\begin{aligned}"
        r"\frac{\partial HI_F}{\partial T_F}={}&"
        r"2.04901523-0.22475541RH"
        r"-2(6.83783\times10^{-3})T_F\\"
        r"&+2(1.22874\times10^{-3})T_F RH"
        r"+8.5282\times10^{-4}RH^2\\"
        r"&-2(1.99\times10^{-6})T_F RH^2"
        r"\end{aligned}"
    )

    st.markdown("Đạo hàm theo độ ẩm tương đối:")
    st.latex(
        r"\begin{aligned}"
        r"\frac{\partial HI_F}{\partial RH}={}&"
        r"10.14333127-0.22475541T_F"
        r"-2(5.481717\times10^{-2})RH\\"
        r"&+1.22874\times10^{-3}T_F^2"
        r"+2(8.5282\times10^{-4})T_F RH\\"
        r"&-2(1.99\times10^{-6})T_F^2 RH"
        r"\end{aligned}"
    )

    st.markdown("""
    Do công thức gốc cho kết quả theo °F nhưng nghiên cứu biểu diễn Heat Index
    theo °C, cần chú ý quy tắc đổi đơn vị khi diễn giải đạo hàm:
    """)
    st.latex(
        r"\frac{\partial HI_C}{\partial T_C}"
        r"=\frac{\partial HI_F}{\partial T_F}"
    )
    st.latex(
        r"\frac{\partial HI_C}{\partial RH}"
        r"=\frac{5}{9}\frac{\partial HI_F}{\partial RH}"
    )

    st.markdown("""
    **Diễn giải:** Đạo hàm theo nhiệt độ có đơn vị °C Heat Index trên mỗi °C
    nhiệt độ. Đạo hàm theo độ ẩm có đơn vị °C Heat Index trên mỗi điểm phần trăm
    độ ẩm tương đối.
    """)

    # --------------------------------------------------------
    # 2.5. KẾT QUẢ VÀ PHÂN TÍCH
    # --------------------------------------------------------
    st.markdown("### 2.5. Kết quả tính toán và phân tích")

    st.markdown("""
    Dựa trên bảng kết quả, có thể nhận thấy:

    - Châu Đốc có nhiệt độ quan trắc cao nhất trong bốn trạm.
    - Châu Đốc cũng có Heat Index cao nhất trong bảng.
    - Giá trị đạo hàm theo độ ẩm tại Châu Đốc là 0,49, cao nhất trong bốn trạm.
    - Giá trị đạo hàm theo nhiệt độ tại Thổ Chu là 2,87, cao nhất trong bốn trạm.

    Điều này cho thấy mức độ nhạy của Heat Index đối với từng yếu tố
    không giống nhau tại mọi điểm quan trắc.
    """)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Heat Index cao nhất", "44,8 °C", "Châu Đốc")
    col_b.metric("∂HI/∂T lớn nhất", "2,87", "Thổ Chu")
    col_c.metric("∂HI/∂RH lớn nhất", "0,49", "Châu Đốc")

    st.markdown("#### So sánh độ nhạy tại các trạm")

    sensitivity_choice = st.radio(
        "Chọn đại lượng cần so sánh:",
        ["Theo nhiệt độ", "Theo độ ẩm"],
        horizontal=True,
        key="heat_sensitivity_choice"
    )

    if sensitivity_choice == "Theo nhiệt độ":
        y_column = "∂HI/∂T"
        chart_title = "Độ nhạy của Heat Index theo nhiệt độ"
        y_label = "∂HI/∂T"
    else:
        y_column = "∂HI/∂RH"
        chart_title = "Độ nhạy của Heat Index theo độ ẩm"
        y_label = "∂HI/∂RH"

    sensitivity_fig = go.Figure()
    sensitivity_fig.add_trace(go.Bar(
        x=weather_data["Trạm khí tượng"],
        y=weather_data[y_column],
        text=weather_data[y_column].map(lambda v: f"{v:.2f}"),
        textposition="outside",
        marker=dict(
            color=["#5B8DB8", "#70A6A0", "#D6A45B", "#B87568"]
        )
    ))
    sensitivity_fig.update_layout(
        title=chart_title,
        xaxis_title="Trạm khí tượng",
        yaxis_title=y_label,
        template="plotly_white",
        height=390,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    st.plotly_chart(sensitivity_fig, use_container_width=True)

    st.markdown("""
    **Cách đọc biểu đồ:** Cột càng cao thì độ nhạy của Heat Index theo biến
    đang xét càng lớn tại điểm quan trắc đó. Đây là độ nhạy cục bộ,
    không có nghĩa biến đó luôn là yếu tố quan trọng nhất trong mọi điều kiện thời tiết.
    """)

    # --------------------------------------------------------
    # 2.6. MINH HỌA TƯƠNG TÁC: TÍNH HEAT INDEX
    # --------------------------------------------------------
    st.markdown("### 2.6. Thử thay đổi nhiệt độ và độ ẩm")

    st.markdown("""
    Có thể thay đổi hai biến đầu vào để quan sát Heat Index thay đổi.
    Công thức dưới đây tính theo phương trình Rothfusz, sau đó chuyển kết quả
    về độ Celsius.
    """)

    def calculate_heat_index_c(temp_c, rh):
        """
        Rothfusz regression.
        Input: temperature in Celsius, RH in percent.
        Output: Heat Index in Celsius.
        """
        t = temp_c * 9 / 5 + 32
        r = rh

        hi_f = (
            -42.379
            + 2.04901523 * t
            + 10.14333127 * r
            - 0.22475541 * t * r
            - 0.00683783 * t**2
            - 0.05481717 * r**2
            + 0.00122874 * t**2 * r
            + 0.00085282 * t * r**2
            - 0.00000199 * t**2 * r**2
        )

        # Điều chỉnh độ ẩm thấp theo công thức Rothfusz
        if r < 13 and 80 <= t <= 112:
            adjustment = (
                ((13 - r) / 4)
                * math.sqrt((17 - abs(t - 95)) / 17)
            )
            hi_f -= adjustment

        # Điều chỉnh độ ẩm cao theo công thức Rothfusz
        elif r > 85 and 80 <= t <= 87:
            adjustment = (
                ((r - 85) / 10)
                * ((87 - t) / 5)
            )
            hi_f += adjustment

        return (hi_f - 32) * 5 / 9

    # math có thể chưa được import trong file gốc
    import math

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        demo_temp = st.slider(
            "Nhiệt độ (°C)",
            min_value=26.0,
            max_value=42.0,
            value=36.1,
            step=0.1,
            key="hi_demo_temp"
        )

    with input_col2:
        demo_rh = st.slider(
            "Độ ẩm tương đối (%)",
            min_value=20,
            max_value=100,
            value=53,
            step=1,
            key="hi_demo_rh"
        )

    demo_hi = calculate_heat_index_c(demo_temp, demo_rh)

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Nhiệt độ", f"{demo_temp:.1f} °C")
    result_col2.metric("Độ ẩm", f"{demo_rh:.0f}%")
    result_col3.metric("Heat Index", f"{demo_hi:.1f} °C")

    st.caption(
        "Lưu ý: Đây là công cụ minh họa phép tính theo phương trình hồi quy. "
        "Phương trình Rothfusz có phạm vi áp dụng và các điều kiện sử dụng riêng; "
        "không nên xem kết quả như đánh giá y tế cá nhân."
    )

    # --------------------------------------------------------
    # 2.7. HÌNH HỌC: MẶT CONG HEAT INDEX
    # --------------------------------------------------------
    st.markdown("### 2.7. Ý nghĩa hình học – Mặt cong Heat Index")

    st.markdown("""
    Ta có thể xem Heat Index là một mặt cong trong không gian ba chiều,
    trong đó hai trục nằm ngang là nhiệt độ và độ ẩm, còn trục đứng là Heat Index.
    """)
    st.latex(r"z=HI(T,RH)")

    st.markdown("""
    Mỗi điểm trên mặt cong tương ứng với một tổ hợp nhiệt độ – độ ẩm.
    Độ cao tại điểm đó biểu diễn chỉ số nhiệt được tính theo mô hình.
    """)

    temp_values = np.linspace(28, 42, 45)
    rh_values = np.linspace(30, 90, 45)
    T_grid, RH_grid = np.meshgrid(temp_values, rh_values)

    HI_grid = np.zeros_like(T_grid, dtype=float)
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            HI_grid[i, j] = calculate_heat_index_c(
                float(T_grid[i, j]),
                float(RH_grid[i, j])
            )

    surface_fig = go.Figure(data=[
        go.Surface(
            x=T_grid,
            y=RH_grid,
            z=HI_grid,
            colorscale="Viridis",
            colorbar=dict(title="HI (°C)")
        )
    ])
    surface_fig.update_layout(
        title="Mặt cong biểu diễn Heat Index theo nhiệt độ và độ ẩm",
        scene=dict(
            xaxis_title="Nhiệt độ (°C)",
            yaxis_title="Độ ẩm (%)",
            zaxis_title="Heat Index (°C)"
        ),
        height=570,
        margin=dict(l=0, r=0, t=55, b=0)
    )
    st.plotly_chart(surface_fig, use_container_width=True)

    st.markdown("""
    **Nhận xét:** Khi di chuyển trên mặt cong theo hướng nhiệt độ tăng
    hoặc độ ẩm tăng, giá trị Heat Index có thể thay đổi.
    Độ dốc của mặt cong tại mỗi điểm được mô tả bởi các đạo hàm riêng.
    """)

    # --------------------------------------------------------
    # 2.8. MẶT PHẲNG TIẾP TUYẾN
    # --------------------------------------------------------
    st.markdown("### 2.8. Mặt phẳng tiếp tuyến và xấp xỉ tuyến tính")

    st.markdown("""
    Tại một điểm \((a,b)\), mặt cong có thể được xấp xỉ bởi mặt phẳng tiếp tuyến.
    Công thức này cho biết Heat Index thay đổi gần đúng như thế nào
    khi nhiệt độ và độ ẩm thay đổi một lượng nhỏ quanh điểm đang xét.
    """)

    st.latex(
        r"\begin{aligned}"
        r"HI(T,RH)\approx{}&HI(a,b)"
        r"+HI_T(a,b)(T-a)\\"
        r"&+HI_{RH}(a,b)(RH-b)"
        r"\end{aligned}"
    )

    st.markdown("""
    Trong đó:
    - \((a,b)\): điểm gốc đang xét.
    - \(HI(a,b)\): Heat Index tại điểm gốc.
    - \(HI_T(a,b)\): đạo hàm riêng theo nhiệt độ tại điểm gốc.
    - \(HI_{RH}(a,b)\): đạo hàm riêng theo độ ẩm tại điểm gốc.

    Mặt phẳng tiếp tuyến chỉ là xấp xỉ cục bộ, thường phù hợp khi xét
    các thay đổi nhỏ quanh điểm gốc.
    """)

    st.markdown("#### Minh họa xấp xỉ tại Châu Đốc")

    # Các giá trị đạo hàm giữ theo bảng kết quả nghiên cứu đã cung cấp.
    base_temp = 36.1
    base_rh = 53.0
    base_hi = 44.8
    base_dT = 2.75
    base_dRH = 0.49

    tangent_temp = st.slider(
        "Nhiệt độ mới (°C)",
        min_value=34.0,
        max_value=39.0,
        value=37.0,
        step=0.1,
        key="hi_tangent_temp"
    )
    tangent_rh = st.slider(
        "Độ ẩm mới (%)",
        min_value=45,
        max_value=65,
        value=56,
        step=1,
        key="hi_tangent_rh"
    )

    tangent_estimate = (
        base_hi
        + base_dT * (tangent_temp - base_temp)
        + base_dRH * (tangent_rh - base_rh)
    )

    exact_estimate = calculate_heat_index_c(tangent_temp, tangent_rh)

    tangent_col1, tangent_col2 = st.columns(2)
    tangent_col1.metric(
        "Xấp xỉ bằng mặt phẳng tiếp tuyến",
        f"{tangent_estimate:.2f} °C"
    )
    tangent_col2.metric(
        "Tính bằng công thức Rothfusz",
        f"{exact_estimate:.2f} °C"
    )

    st.markdown("""
    **Phân tích:** Hai kết quả có thể không trùng nhau vì mặt phẳng tiếp tuyến
    chỉ xấp xỉ mặt cong trong vùng lân cận điểm gốc. Khi điểm mới càng xa
    điểm gốc, sai số xấp xỉ có thể càng đáng kể.

    Lưu ý: phần minh họa này sử dụng các đạo hàm tại Châu Đốc trong bảng kết quả
    nghiên cứu đã cung cấp. Cần bảo đảm các đạo hàm đó được tính từ cùng một
    quy ước đơn vị và cùng công thức trước khi dùng để đối chiếu định lượng.
    """)

    # --------------------------------------------------------
    # 2.9. ĐƯỜNG ĐỒNG MỨC
    # --------------------------------------------------------
    st.markdown("### 2.9. Đường đồng mức – Những điều kiện thời tiết có cùng Heat Index")

    st.markdown("""
    Đường đồng mức là tập hợp những điểm có cùng giá trị Heat Index:
    """)
    st.latex(r"HI(T,RH)=c")

    st.markdown("""
    Trên mặt phẳng nhiệt độ – độ ẩm, mỗi đường biểu diễn các tổ hợp
    nhiệt độ và độ ẩm tạo ra cùng một chỉ số nhiệt.
    """)

    contour_fig = go.Figure(data=go.Contour(
        x=temp_values,
        y=rh_values,
        z=HI_grid,
        colorscale="RdYlBu_r",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=11, color="black")
        ),
        colorbar=dict(title="HI (°C)")
    ))

    contour_fig.add_trace(go.Scatter(
        x=[base_temp],
        y=[base_rh],
        mode="markers+text",
        marker=dict(size=12, color="black", symbol="x"),
        text=["Châu Đốc"],
        textposition="top right",
        name="Điểm Châu Đốc"
    ))

    contour_fig.update_layout(
        title="Đường đồng mức Heat Index theo nhiệt độ và độ ẩm",
        xaxis_title="Nhiệt độ (°C)",
        yaxis_title="Độ ẩm tương đối (%)",
        height=520,
        template="plotly_white"
    )
    st.plotly_chart(contour_fig, use_container_width=True)

    st.markdown("""
    **Ý nghĩa thực tế:** Hai ngày có nhiệt độ và độ ẩm khác nhau vẫn có thể
    tạo ra cùng một mức Heat Index. Đường đồng mức giúp quan sát những tổ hợp
    điều kiện thời tiết tương đương về chỉ số nhiệt.
    """)

    # --------------------------------------------------------
    # 2.10. Ý NGHĨA VÀ GIỚI HẠN
    # --------------------------------------------------------
    st.markdown("### 2.10. Ý nghĩa thực tiễn")

    meaning_col1, meaning_col2 = st.columns(2)

    with meaning_col1:
        st.markdown("""
        **Đối với phân tích khí tượng**
        - Kết hợp nhiệt độ và độ ẩm để đánh giá mức nóng cảm nhận.
        - So sánh mức độ nóng giữa các trạm.
        - Phân tích yếu tố nào làm Heat Index nhạy hơn tại từng điểm.
        """)

    with meaning_col2:
        st.markdown("""
        **Đối với cảnh báo nắng nóng**
        - Hỗ trợ truyền tải thông tin về mức nóng cảm nhận.
        - Giúp người dân chú ý hơn đến điều kiện nóng ẩm.
        - Là một nguồn thông tin để hỗ trợ khuyến cáo phòng tránh nắng nóng.
        """)

    with st.expander("Giới hạn của mô hình và những điều cần lưu ý"):
        st.markdown("""
        1. **Mô hình thực nghiệm:** Rothfusz là phương trình hồi quy được xây dựng
        từ dữ liệu thực nghiệm, không phải mô hình mô tả đầy đủ mọi cơ chế trao đổi
        nhiệt của cơ thể người.

        2. **Phạm vi áp dụng:** Không nên sử dụng công thức ngoài phạm vi và
        điều kiện khuyến nghị của cơ quan khí tượng.

        3. **Các yếu tố khác:** Cảm giác nóng còn phụ thuộc vào bức xạ mặt trời,
        tốc độ gió, quần áo, hoạt động thể chất, thời gian phơi nhiễm và đặc điểm
        của từng cá nhân.

        4. **Ý nghĩa đạo hàm:** Đạo hàm riêng thể hiện độ nhạy cục bộ tại một điểm,
        không phải mức thay đổi cố định cho mọi điều kiện.

        5. **Đơn vị:** Phải thống nhất °F, °C và cách biểu diễn độ ẩm khi tính
        đạo hàm. Đặc biệt, đạo hàm theo RH phải được đổi đơn vị nhất quán
        với cách biểu diễn Heat Index.
        """)

    # --------------------------------------------------------
    # KẾT LUẬN
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## KẾT LUẬN")

    st.markdown("""
    Qua nghiên cứu chỉ số nhiệt Heat Index, ta thấy đạo hàm riêng không chỉ
    là một phép tính toán học mà còn là công cụ để phân tích mức độ ảnh hưởng
    của từng yếu tố trong một hiện tượng thực tế.

    - Mô hình \(HI=f(T,RH)\) biểu diễn mối quan hệ giữa nhiệt độ, độ ẩm và chỉ số nhiệt.
    - Đạo hàm riêng theo nhiệt độ và độ ẩm cho biết độ nhạy của Heat Index đối với từng yếu tố.
    - Gradient cho biết hướng mà Heat Index tăng nhanh nhất.
    - Mặt phẳng tiếp tuyến giúp xấp xỉ sự thay đổi của Heat Index trong vùng lân cận một điểm.
    - Đường đồng mức giúp so sánh các tổ hợp nhiệt độ và độ ẩm tạo ra cùng chỉ số nhiệt.

    **Từ một mô hình nhiều biến, đạo hàm riêng giúp chuyển những thay đổi
    của từng yếu tố thành thông tin định lượng, qua đó hỗ trợ phân tích
    và ra quyết định trong thực tiễn.**
    """)

    # --------------------------------------------------------
    # TÀI LIỆU THAM KHẢO
    # --------------------------------------------------------
    with st.expander("Tài liệu tham khảo"):
        st.markdown("""
        **1. Heat Index và khí tượng học**
        - [NOAA / National Weather Service – Heat Index Equation](https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml)
        - [National Weather Service – Heat Index](https://www.weather.gov/safety/heat-index)

        **2. Kiến trúc, quy hoạch và GIS**
        - [ArcGIS – How Slope Works](https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm)
        - [Google Earth Engine – Terrain Slope](https://developers.google.com/earth-engine/apidocs/ee-terrain-slope)

        **3. Vi sinh vật học**
        - [Monod – The Growth of Bacterial Cultures](https://doi.org/10.1146/annurev.mi.03.100149.002103)

        **4. Kinh tế học**
        - [Econlib – Production Theory](https://www.econlib.org/library/Enc/ProductionTheory.html)

        **5. Trí tuệ nhân tạo**
        - [Deep Learning Book – Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)

        **6. Xử lý ảnh**
        - [Springer – Image Processing](https://link.springer.com/chapter/10.1007/978-3-030-01364-6_3)
        """)
st.divider()
st.caption("Lưu ý: các kết quả đạo hàm được tính ký hiệu bằng SymPy; đồ thị và nghiệm điểm tới hạn là biểu diễn/tính toán số trong miền khảo sát.")
