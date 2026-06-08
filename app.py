import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

# =====================================================================
# 1. Streamlit 웹 페이지 기본 설정
# =====================================================================
st.set_page_config(page_title="Professional Equity Research Agent", layout="wide")
st.title("📊 수석 에퀴티 리서치 애널리스트 에이전트")
st.caption("계량금융학 및 비정형 데이터 분석 기반 종합 주식 가치평가 시스템")

# 사이드바에서 API 키 및 티커 입력 받기
st.sidebar.header("설정 (Settings)")
api_key_input = st.sidebar.text_input("Gemini API Key 입력", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
target_ticker = st.sidebar.text_input("분석할 기업 Ticker 입력 (예: AAPL, TSLA, 005930.KS)", value="AAPL").strip()

# =====================================================================
# 2. 데이터 수집 및 계량 연산 파이프라인
# =====================================================================
def get_financial_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    
    baseline = {
        "최신 주가": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
        "시가총액": info.get("marketCap", "N/A"),
        "총 발행 주식 수": info.get("sharesOutstanding", "N/A"),
        "주당순이익(EPS)": info.get("trailingEps", "N/A"),
        "52주 최고가": info.get("fiftyTwoWeekHigh", "N/A"),
        "52주 최저가": info.get("fiftyTwoWeekLow", "N/A"),
        "최근 거래량": info.get("volume", "N/A"),
    }
    
    # 뉴스 데이터 수집
    news_stream = ticker.news
    text_corpus = ""
    if news_stream:
        for idx, art in enumerate(news_stream[:5]):
            text_corpus += f"[{idx+1}] {art['title']} / {art.get('summary', '')}\n"
            
    return baseline, text_corpus

def run_monte_carlo(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1y")
    if hist.empty: return "데이터 부족"
    
    returns = hist['Close'].pct_change().dropna()
    drift = returns.mean() - (0.5 * returns.var())
    std_dev = returns.std()
    
    last_price = hist['Close'].iloc[-1]
    # 252영업일, 500회 시뮬레이션
    daily_returns = np.exp(drift + std_dev * np.random.normal(0, 1, (252, 500)))
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    for t in range(1, 252):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return {
        "current": last_price,
        "mean": np.mean(price_paths[-1]),
        "upper": np.percentile(price_paths[-1], 95),
        "lower": np.percentile(price_paths[-1], 5)
    }

# =====================================================================
# 3. 에이전트 실행 버튼 및 추론 엔진 (교수님 라이브러리 규격)
# =====================================================================
if st.sidebar.button("전사적 정밀 분석 시작"):
    if not api_key_input:
        st.error("오류: Gemini API Key를 입력해주세요.")
    else:
        # 교수님 라이브러리(google-generativeai) 설정 방식
        genai.configure(api_key=api_key_input)
        
        with st.spinner("금융 데이터를 수집하고 인공지능 에이전트가 보고서를 작성 중입니다..."):
            try:
                # 데이터 수집 실행
                baseline, news = get_financial_data(target_ticker)
                mc = run_monte_carlo(target_ticker)
                
                # 대시보드상에 정량 데이터 먼저 시각화 (Streamlit 기능 활용)
                st.subheader(f"📈 {target_ticker} 실시간 정량 지표 스냅샷")
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 주가", f"${baseline['최신 주가']}" if isinstance(baseline['최신 주가'], (int, float)) else "N/A")
                col2.metric("1년 후 통계적 예상가", f"${mc['mean']:.2f}" if isinstance(mc, dict) else "N/A")
                col3.metric("52주 최고가", f"${baseline['52주 최고가']}" if isinstance(baseline['52주 최고가'], (int, float)) else "N/A")
                
                # 프롬프트 조립
                prompt = f"""
## 실시간 금융 데이터 소스
1. 기초 데이터: {baseline}
2. 몬테카를로 가치평가: {mc}
3. 비정형 뉴스 텍스트: {news}

위 데이터를 바탕으로 제공된 System Instructions(수석 에퀴티 애널리스트 페르소나, Step 1~8 프로세스, LaTeX 수식 및 Excel 수식 병기, 통합 스코어카드 포함)를 엄격히 준수하여 최고급 종합 주식 평가 보고서를 작성해줘.
"""
                
                # 교수님 라이브러리 모델 호출 방식
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                
                # 결과 출력
                st.markdown("---")
                st.subheader("📋 수석 애널리스트 최종 리서치 보고서")
                st.markdown(response.text)
                
                # 다운로드 버튼 기능 제공
                st.download_button(
                    label="📥 보고서 마크다운 파일 다운로드",
                    data=response.text,
                    file_name=f"Equity_Report_{target_ticker}_{datetime.date.today()}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
