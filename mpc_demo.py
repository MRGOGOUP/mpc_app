import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import streamlit as st
import cvxpy as cp
import platform

# ✅ 한글 폰트 설정 (Windows 11 기준)
matplotlib.rc( 'font', family = 'Malgun Gothic')
#matplotlib.rcParams['font.family'] = 'Malgun Gothic'
#matplotlib.rcParams['axes.unicode_minus'] = False

# 📋 Streamlit UI 설정
st.set_page_config(page_title="APC 시뮬레이션 (DCS 기반)", layout="centered")
st.title("🏭 DCS + APC 시뮬레이션 데모")

st.markdown("""
이 시뮬레이션은 **비철금속 공장의 DCS 제어 환경**을 가정하고,  
여기에 **APC(Model Predictive Control)**를 적용한 예시입니다.

---

### ⚙️ 사용된 제어 수식 (MPC 최적화 목적 함수):

$$
\\min_{u_0, \\dots, u_{N-1}} \\sum_{k=1}^{N} (y_k - y_{\\text{set}})^2
$$

- $y_k$: 예측된 공정 출력값
- $y_{\\text{set}}$: 목표값(Setpoint)
- $N$: 예측 구간 (Prediction Horizon)

---

APC는 미래의 공정 동작을 **모델로 예측**하고,  
해당 예측값들이 목표값에 가장 가까워지도록 **최적의 제어 입력 시퀀스**를 계산합니다.
""", unsafe_allow_html=True)

# 📘 설명 추가
with st.expander("📖 APC 제어 방식 설명"):
    st.markdown("""
    - **DCS (Distributed Control System)**는 공장의 센서 및 제어 신호를 실시간으로 처리하는 기반 시스템입니다.
    - **APC (Advanced Process Control)**는 DCS 위에서 더 정교한 알고리즘(예: 예측제어)을 실행하여 품질과 생산성을 향상시킵니다.
    - 이 시뮬레이션은 단순 1차 공정 모델을 기반으로 MPC를 구현한 것입니다.
    """)

# 기존 코드와 함께 아래 부분을 추가하세요 (그래프와 표 출력 이후 위치 추천)

# 📘 공정 모델 설명
with st.expander("📖 공정 모델 설명 (현재 사용 중인 y[k+1] = a*y[k] + b*u[k])"):
    st.markdown(r"""
이 시뮬레이터는 공정을 단순한 **1차 시스템**으로 가정하고 있습니다:

---

### ⚙️ 공정 수식:

$$
y[k+1] = a \cdot y[k] + b \cdot u[k]
$$

- $y[k]$: 현재 공정 출력 (예: 온도, 농도, 압력 등)  
- $u[k]$: 현재 제어 입력 (예: 밸브 개도율, 가열량 등)  
- $a$: 관성 계수 – 과거 출력이 현재에 얼마나 영향을 주는가  
- $b$: 민감도 계수 – 입력이 출력에 얼마나 영향을 주는가  

---

### 🔧 예시 해석:

예: $a = 0.85$, $b = 0.5$

- 출력값은 과거의 85% 정도를 유지하면서  
- 제어 입력에 의해 50%만큼 변화  
→ **천천히 반응하는 안정적인 시스템**

---

### 🏭 실제 공정 적용 예 (비철금속 공장):

| 항목 | 예시 |
|------|------|
| 출력값 $y$ | 정제조의 온도 또는 농도 |
| 제어입력 $u$ | 스팀 밸브 조작, 첨가제 투입량 |
| 외란 | 외부 온도 변화, 원료 농도 편차 |

---

### 📈 실제 적용 시에는?

- 과거 DCS 데이터를 기반으로 $a$, $b$ 추정 (회귀 또는 시스템 식별)  
- 모델 예측을 통해 최적의 $u$를 구하고 APC에 적용  
- 다중 입력/출력(MIMO), 시간 지연, 외란 보상 등을 포함한 **복잡한 모델**로 확장 가능
    """, unsafe_allow_html=True)

    
# 🎛 사용자 설정값 (사이드바로 이동)
with st.sidebar:
    st.header("🔧 제어 파라미터 설정")
    setpoint = st.slider("🎯 공정 목표값", 0.0, 2.0, 1.0, 0.1)
    a = st.slider("📉 공정 반응 계수 a", 0.0, 1.0, 0.85, 0.01)
    b = st.slider("⚙️ 제어 민감도 계수 b", 0.1, 2.0, 0.5, 0.1)
    N = st.slider("⏱ APC 예측 구간 (N)", 2, 20, 10)
    horizon = st.slider("🕒 시뮬레이션 시간 (초)", 10, 60, 30)

# 📌 제약조건
u_min, u_max = -1.0, 1.0
y_min, y_max = -0.3, 1.6

# 🔁 시뮬레이션 시작
y = 0
y_history = [y]
u_history = []

for k in range(horizon):
    u = cp.Variable(N)
    y_pred = []
    y_next = y

    for i in range(N):
        y_next = a * y_next + b * u[i]
        y_pred.append(y_next)

    cost = cp.sum_squares(cp.hstack(y_pred) - setpoint)
    constraints = [u >= u_min, u <= u_max]
    constraints += [yp >= y_min for yp in y_pred]
    constraints += [yp <= y_max for yp in y_pred]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    u_val = u.value[0]
    y = a * y + b * u_val

    y_history.append(y)
    u_history.append(u_val)

# 📈 그래프 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
time = np.arange(horizon + 1)

# 출력값 그래프
ax1.plot(time, y_history, label="공정 출력값 (PV)")
ax1.plot(time, [setpoint]*len(time), 'r--', label="목표값 (Setpoint)")
ax1.axhline(y_min, color='gray', linestyle='--', linewidth=1, label="출력 제약")
ax1.axhline(y_max, color='gray', linestyle='--', linewidth=1)
ax1.set_ylabel("출력값 (예: 온도)")
ax1.grid(True)
ax1.legend()

# 제어입력 그래프
ax2.step(np.arange(horizon), u_history, label="제어 입력 (MV)")
ax2.axhline(u_min, color='gray', linestyle='--', linewidth=1, label="입력 제약")
ax2.axhline(u_max, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel("시간")
ax2.set_ylabel("입력값")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

# ✅ 결과 표 형태로도 출력
import pandas as pd

result_df = pd.DataFrame({
    "시간": np.arange(horizon + 1),
    "출력값 (PV)": y_history,
    "입력값 (MV)": [u_history[0]] + u_history  # MV는 1개 부족하므로 보정
})

st.subheader("📋 입력값(MV) 및 출력값(PV) 요약")
st.dataframe(result_df.style.format(precision=3), use_container_width=True)



