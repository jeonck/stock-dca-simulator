import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="적립식 투자 수익률 계산기", layout="wide")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("투자 설정")

    ticker = st.text_input("티커", value="QLD").upper()

    today = datetime.today().date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", today - timedelta(days=365))
    with col2:
        end_date = st.date_input("종료일", today)

    frequency = st.selectbox(
        "매수 주기",
        options=["매일", "매주 (월요일)", "격주 (월요일)", "매월 (1영업일)"],
        index=0,
    )

    shares_per_buy = st.number_input("회당 매수 주수", min_value=1, value=1, step=1)

# 티커 정보로 타이틀 생성
ticker_info = yf.Ticker(ticker)
long_name = ticker_info.info.get("longName", ticker)
st.title(f"{ticker} 적립식 투자 수익률 계산기")
st.caption(f"{long_name} - 적립식 투자 시뮬레이션")


# --- 데이터 수집 ---
@st.cache_data(ttl=3600, show_spinner="주가 데이터 불러오는 중...")
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # yfinance end date is exclusive, add 1 day
    end_dt = pd.to_datetime(end) + timedelta(days=1)
    df = yf.download(ticker, start=start, end=end_dt.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


df = load_data(ticker, str(start_date), str(end_date))

if df.empty:
    st.error(f"'{ticker}' 데이터를 불러올 수 없습니다. 티커와 날짜를 확인해주세요.")
    st.stop()

# --- 매수일 필터링 ---
df = df.copy()
df.index = pd.to_datetime(df.index)

if frequency == "매일":
    buy_dates = df.index
elif frequency == "매주 (월요일)":
    buy_dates = df.index[df.index.weekday == 0]
elif frequency == "격주 (월요일)":
    mondays = df.index[df.index.weekday == 0]
    buy_dates = mondays[::2]
else:  # 매월
    buy_dates = df.groupby(df.index.to_period("M")).apply(lambda x: x.index.min()).values
    buy_dates = pd.DatetimeIndex(buy_dates)

# --- 시뮬레이션 ---
records = []
total_shares = 0
total_invested = 0.0

for date in df.index:
    close_price = float(df.loc[date, "Close"])
    bought = False
    if date in buy_dates:
        total_shares += shares_per_buy
        total_invested += close_price * shares_per_buy
        bought = True

    current_value = total_shares * close_price
    profit = current_value - total_invested
    profit_pct = (profit / total_invested * 100) if total_invested > 0 else 0.0

    records.append(
        {
            "날짜": date,
            "종가": close_price,
            "매수": bought,
            "누적주수": total_shares,
            "총투자금": total_invested,
            "평가금액": current_value,
            "수익금": profit,
            "수익률(%)": profit_pct,
        }
    )

result = pd.DataFrame(records).set_index("날짜")

# --- 요약 지표 ---
st.divider()
last = result.iloc[-1]
avg_price = last["총투자금"] / last["누적주수"] if last["누적주수"] > 0 else 0
total_buys = result["매수"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 매수 횟수", f"{int(total_buys)}회")
c2.metric("누적 주수", f"{int(last['누적주수'])}주")
c3.metric("총 투자금", f"${last['총투자금']:,.2f}")
c4.metric("현재 평가금", f"${last['평가금액']:,.2f}")
c5.metric(
    "수익률",
    f"{last['수익률(%)']:+.2f}%",
    delta=f"${last['수익금']:+,.2f}",
    delta_color="normal",
)

st.metric("평균 매수 단가", f"${avg_price:,.2f} (현재가 ${last['종가']:,.2f})")

# --- 그래프 1: 투자금 vs 평가금 ---
st.divider()
st.subheader("투자금 vs 평가금액 추이")

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=result.index,
        y=result["총투자금"],
        name="총 투자금",
        line=dict(color="#888888", dash="dot"),
        fill=None,
    )
)
fig1.add_trace(
    go.Scatter(
        x=result.index,
        y=result["평가금액"],
        name="평가금액",
        line=dict(color="#2196F3"),
        fill="tonexty",
        fillcolor="rgba(33,150,243,0.15)",
    )
)
fig1.update_layout(
    height=420,
    yaxis_title="USD ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig1, use_container_width=True)

# --- 그래프 2: 수익률 추이 ---
st.subheader("수익률 추이")

colors = ["#4CAF50" if v >= 0 else "#F44336" for v in result["수익률(%)"]]
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=result.index,
        y=result["수익률(%)"],
        name="수익률",
        line=dict(color="#FF9800", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.12)",
    )
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(
    height=350,
    yaxis_title="수익률 (%)",
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True)

# --- 그래프 3: 주가 + 매수 시점 ---
st.subheader("주가 차트 & 매수 시점")

buy_points = result[result["매수"]]
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=result.index,
        y=result["종가"],
        name="종가",
        line=dict(color="#607D8B"),
    )
)
fig3.add_trace(
    go.Scatter(
        x=buy_points.index,
        y=buy_points["종가"],
        mode="markers",
        name="매수 시점",
        marker=dict(color="#4CAF50", size=5, symbol="triangle-up"),
    )
)
fig3.add_hline(
    y=avg_price,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"평균단가 ${avg_price:.2f}",
)
fig3.update_layout(
    height=380,
    yaxis_title="USD ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig3, use_container_width=True)

# --- 복리 효과 비교: 적립식 vs 일시불 ---
st.divider()
st.subheader("적립식 vs 일시불 투자 비교 (복리 효과)")

first_price = float(df.iloc[0]["Close"])
lump_sum_shares = total_invested / first_price  # 같은 금액을 첫날에 모두 투입

result["일시불_평가금"] = lump_sum_shares * result["종가"]
result["일시불_수익률(%)"] = (result["일시불_평가금"] - total_invested) / total_invested * 100

fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(
        x=result.index,
        y=result["수익률(%)"],
        name="적립식 수익률",
        line=dict(color="#2196F3", width=2),
    )
)
fig4.add_trace(
    go.Scatter(
        x=result.index,
        y=result["일시불_수익률(%)"],
        name="일시불 수익률",
        line=dict(color="#F44336", width=2, dash="dash"),
    )
)
fig4.add_hline(y=0, line_dash="dot", line_color="gray")
fig4.update_layout(
    height=400,
    yaxis_title="수익률 (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig4, use_container_width=True)

lump_last = result["일시불_수익률(%)"].iloc[-1]
st.info(
    f"같은 금액(${total_invested:,.2f})을 **{start_date}에 일시불** 투자했다면 "
    f"수익률: **{lump_last:+.2f}%** | "
    f"적립식 수익률: **{last['수익률(%)']:+.2f}%**"
)

# --- 상세 데이터 ---
with st.expander("상세 매매 데이터 보기"):
    display_df = result[result["매수"]].copy()
    display_df = display_df[["종가", "누적주수", "총투자금", "평가금액", "수익금", "수익률(%)"]]
    display_df.columns = ["매수가", "누적주수", "총투자금($)", "평가금($)", "수익금($)", "수익률(%)"]
    st.dataframe(
        display_df.style.format(
            {
                "매수가": "${:,.2f}",
                "총투자금($)": "${:,.2f}",
                "평가금($)": "${:,.2f}",
                "수익금($)": "${:+,.2f}",
                "수익률(%)": "{:+.2f}%",
            }
        ),
        use_container_width=True,
    )
