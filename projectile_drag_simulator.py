import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="포물선 운동 최적화 시뮬레이터", layout="wide")
st.title("공기 저항에 따른 최적 발사 각도")
st.caption("공기 저항을 고려한 포물선 운동을 수치적으로 계산합니다. 실제 실험이 아닌 수학적 모델 기반 시뮬레이션입니다.")

st.sidebar.header("시뮬레이션 조건")
v0 = st.sidebar.slider("초기 속력 (m/s)", 5.0, 40.0, 20.0, 1.0)
mass = st.sidebar.slider("물체의 질량 (kg)", 0.1, 2.0, 0.5, 0.1)
g = st.sidebar.slider("중력 가속도 (m/s²)", 1.0, 15.0, 9.8, 0.1)
k = st.sidebar.slider("공기 저항 계수 k (kg/m)", 0.0, 0.10, 0.02, 0.005)
dt = st.sidebar.select_slider("계산 시간 간격 (s)", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.005)

st.sidebar.markdown(
    "공기 저항력 모델: **Fᵈ = −k |v| v**  \n"
    "k가 클수록 저항이 커집니다. 이 모델에서 k는 질량과 별개로 조절하는 유효 계수입니다."
)

def simulate(angle_deg, v0, mass, g, k, dt, max_time=30.0):
    """Quadratic drag model; semi-implicit Euler integration."""
    angle = np.deg2rad(angle_deg)
    x, y = 0.0, 0.0
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    xs, ys = [x], [y]
    t = 0.0

    while t < max_time:
        speed = np.hypot(vx, vy)
        ax = -(k / mass) * speed * vx
        ay = -g - (k / mass) * speed * vy

        old_x, old_y = x, y
        old_vy = vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < 0:
            # Interpolate the final ground crossing for a more accurate range.
            denom = old_y - y
            frac = old_y / denom if denom != 0 else 0.0
            landing_x = old_x + frac * (x - old_x)
            xs.append(landing_x)
            ys.append(0.0)
            return np.array(xs), np.array(ys), landing_x, t

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys), x, t

angles = np.arange(5.0, 86.0, 1.0)
ranges = []
for a in angles:
    _, _, r, _ = simulate(a, v0, mass, g, k, dt)
    ranges.append(r)
ranges = np.array(ranges)
best_idx = int(np.argmax(ranges))
best_angle = float(angles[best_idx])
best_range = float(ranges[best_idx])

# No-drag theoretical comparison
ideal_angle = 45.0
ideal_range = v0**2 / g

col1, col2, col3 = st.columns(3)
col1.metric("최적 발사 각도 (현재 조건)", f"{best_angle:.0f}°")
col2.metric("최대 사거리 (현재 조건)", f"{best_range:.2f} m")
col3.metric("공기 저항이 없을 때의 이론 사거리", f"{ideal_range:.2f} m", "45° 발사 가정")

left, right = st.columns(2)

with left:
    st.subheader("1. 발사 각도별 궤적")
    selected_angle = st.slider("궤적을 자세히 볼 발사 각도 (°)", 5, 85, int(best_angle), 1)
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    for a in [selected_angle, 30, 45, 60]:
        xs, ys, _, _ = simulate(a, v0, mass, g, k, dt)
        ax1.plot(xs, ys, label=f"{a}°")
    ax1.set_xlabel("수평 거리 x (m)")
    ax1.set_ylabel("높이 y (m)")
    ax1.set_title("공기 저항을 고려한 궤적")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

with right:
    st.subheader("2. 발사 각도와 사거리")
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(angles, ranges, label="공기 저항 고려")
    ax2.scatter([best_angle], [best_range], zorder=3, label=f"최대: {best_angle:.0f}°")
    ax2.axvline(45, linestyle="--", alpha=0.7, label="45° 기준선")
    ax2.set_xlabel("발사 각도 (°)")
    ax2.set_ylabel("수평 사거리 (m)")
    ax2.set_title("발사 각도에 따른 사거리")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

st.subheader("3. 조건 비교")
st.write(
    f"현재 조건에서 공기 저항을 고려했을 때 계산된 최적 발사 각도는 **{best_angle:.0f}°**, "
    f"최대 사거리는 **{best_range:.2f} m**입니다. "
    f"공기 저항이 없는 이상적인 모델에서는 같은 높이에서 발사·착지할 때 "
    f"45°에서 사거리가 최대이며, 사거리는 **{ideal_range:.2f} m**입니다."
)

st.info(
    "탐구 팁: 한 번에 한 변수만 바꾸세요. 먼저 k를 0으로 두고 기준 결과를 확인한 뒤, "
    "k를 0.01, 0.02, 0.04, 0.06 kg/m 등으로 바꾸어 최적 각도와 최대 사거리의 변화를 기록하면 됩니다. "
    "이 결과는 선택한 저항 모델과 조건에 따른 것이며, 모든 물체에 적용되는 보편적인 각도는 아닙니다."
)

with st.expander("모델의 가정과 한계"):
    st.markdown(
        "- 발사 높이와 착지 높이가 같습니다.\n"
        "- 중력 가속도는 일정합니다.\n"
        "- 공기 저항은 속력의 제곱에 비례하는 형태로 근사합니다: Fᵈ = −k|v|v.\n"
        "- 회전, 바람, 양력, 물체의 모양 변화는 고려하지 않습니다.\n"
        "- 운동 방정식은 작은 시간 간격으로 수치적으로 계산하므로 시간 간격에 따라 작은 오차가 생길 수 있습니다."
    )
