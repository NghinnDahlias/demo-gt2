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

tab_calc, tab_geometry, tab_change = st.tabs([
    "1 · TÍNH ĐẠO HÀM RIÊNG",
    "2 · Ý NGHĨA HÌNH HỌC",
    "3 · SỰ BIẾN THIÊN"
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

st.divider()
st.caption("Lưu ý: các kết quả đạo hàm được tính ký hiệu bằng SymPy; đồ thị và nghiệm điểm tới hạn là biểu diễn/tính toán số trong miền khảo sát.")
