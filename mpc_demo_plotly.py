import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cvxpy as cp

# 페이지 설정
st.set_page_config(page_title="APC 시뮬레이션 (DCS 기반)", layout="centered")
st.title("🏭 DCS + APC 시뮬레이션 데모 - Plotly 시각화")

# 사이드바 제어 파라미터
with st.sidebar:
    st.header("🔧 제어 파라미터 설정")
    setpoint = st.slider("🎯 공정 목표값", 0.0, 2.0, 1.0, 0.1)
    a = st.slider("📉 공정 반응 계수 a", 0.0, 1.0, 0.85, 0.01)
    b = st.slider("⚙️ 제어 민감도 계수 b", 0.1, 2.0, 0.5, 0.1)
    N = st.slider("⏱ APC 예측 구간 (N)", 2, 20, 10)
    horizon = st.slider("🕒 시뮬레이션 시간 (초)", 10, 60, 30)

# 제약조건
u_min, u_max = -1.0, 1.0
y_min, y_max = -0.3, 1.6

# 시뮬레이션
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

# 시간축
time = np.arange(horizon + 1)

# 📊 Plotly - 출력값(PV)
fig_y = go.Figure()
fig_y.add_trace(go.Scatter(x=time, y=y_history, mode='lines+markers', name="공정 출력값 (PV)"))
fig_y.add_trace(go.Scatter(x=time, y=[setpoint]*len(time), mode='lines', name="목표값 (Setpoint)", line=dict(dash='dash', color='red')))
fig_y.add_trace(go.Scatter(x=time, y=[y_min]*len(time), mode='lines', name="출력 하한", line=dict(dash='dot', color='gray')))
fig_y.add_trace(go.Scatter(x=time, y=[y_max]*len(time), mode='lines', name="출력 상한", line=dict(dash='dot', color='gray')))
fig_y.update_layout(title="📈 공정 출력값(PV)", xaxis_title="시간", yaxis_title="출력값", legend_title="범례", height=400)
st.plotly_chart(fig_y, use_container_width=True)

# 📊 Plotly - 제어입력(MV)
fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=np.arange(horizon), y=u_history, mode='lines+markers', name="제어 입력 (MV)"))
fig_u.add_trace(go.Scatter(x=np.arange(horizon), y=[u_min]*horizon, mode='lines', name="입력 하한", line=dict(dash='dot', color='gray')))
fig_u.add_trace(go.Scatter(x=np.arange(horizon), y=[u_max]*horizon, mode='lines', name="입력 상한", line=dict(dash='dot', color='gray')))
fig_u.update_layout(title="⚙️ 제어 입력값(MV)", xaxis_title="시간", yaxis_title="입력값", legend_title="범례", height=400)
st.plotly_chart(fig_u, use_container_width=True)

# 📋 표 출력
result_df = pd.DataFrame({
    "시간": time,
    "출력값 (PV)": y_history,
    "입력값 (MV)": [u_history[0]] + u_history
})
st.subheader("📋 입력값(MV) 및 출력값(PV) 요약")
st.dataframe(result_df.style.format(precision=3), use_container_width=True)
