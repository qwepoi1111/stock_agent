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
st.caption("계량금융학 및 비정형 데이터 분석 기반 종합 주식 가치평가 시스템 (국내/해외 주식 예외처리 완비)")

# 사이드바에서 API 키 및 티커 입력 받기
st.sidebar.header("설정 (Settings)")
api_key_input = st.sidebar.text_input("Gemini API Key 입력", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
target_ticker = st.sidebar.text_input("분석할 기업 Ticker 입력 (예: AAPL, TSLA, 000660.KS)", value="000660.KS").strip()

# =====================================================================
# 2. 데이터 수집 및 계량 연산 파이프라인 (안전 보강 버전)
# =====================================================================
def get_financial_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
    except Exception:
        info = {}
    
    # 기본 데이터 바인딩 (데이터가 없을 경우 'N/A' 안전 처리)
    baseline = {
        "최신 주가": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
        "시가총액": info.get("marketCap", "N/A"),
        "총 발행 주식 수": info.get("sharesOutstanding", "N/A"),
        "주당순이익(EPS)": info.get("trailingEps", "N/A"),
        "52주 최고가": info.get("fiftyTwoWeekHigh", "N/A"),
        "52주 최저가": info.get("fiftyTwoWeekLow", "N/A"),
        "최근 거래량": info.get("volume", "N/A"),
    }
    
    # [💡 버그 수정 포인트] 뉴스 데이터 수집 시 발생하던 KeyError: 'title' 완전 방어
    try:
        news_stream = ticker.news
    except Exception:
        news_stream = []
        
    text_corpus = ""
    if news_stream:
        for idx, art in enumerate(news_stream[:5]):
            # art['title'] 대신 .get()을 사용하여 키가 없으면 '제목 없음'으로 대체
            title = art.get('title', '관련 뉴스 제목 없음')
            summary = art.get('summary', art.get('description', '본문 요약 내용 없음'))
            text_corpus += f"[{idx+1}] {title} / {summary}\n"
    else:
        text_corpus = "해당 티커에 대한 실시간 수집 뉴스 텍스트가 존재하지 않습니다."
            
    return baseline, text_corpus

def run_monte_carlo(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1y")
    if hist.empty or len(hist) < 10: 
        return "데이터 부족"
    
    returns = hist['Close'].pct_change().dropna()
    if returns.empty:
        return "수익률 연산 불가"
        
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
# 3. 에이전트 실행 버튼 및 추론 엔진
# =====================================================================
if st.sidebar.button("전사적 정밀 분석 시작"):
    if not api_key_input:
        st.error("오류: Gemini API Key를 입력해주세요.")
    else:
        genai.configure(api_key=api_key_input)
        
        with st.spinner(f"{target_ticker} 데이터를 정밀 분석 중입니다. 잠시만 기다려주세요..."):
            try:
                # 데이터 수집 및 시뮬레이션 가동
                baseline, news = get_financial_data(target_ticker)
                mc = run_monte_carlo(target_ticker)
                
                # 대시보드 리포팅
                st.subheader(f"📈 {target_ticker} 실시간 정량 지표 스냅샷")
                col1, col2, col3 = st.columns(3)
                
                cur_price = baseline['최신 주가']
                col1.metric("현재 주가", f"{cur_price:,.0f}원" if isinstance(cur_price, (int, float)) and "KS" in target_ticker else f"${cur_price}" if isinstance(cur_price, (int, float)) else "N/A")
                
                if isinstance(mc, dict):
                    col2.metric("1년 후 통계적 예상가 (평균)", f"{mc['mean']:,.0f}원" if "KS" in target_ticker else f"${mc['mean']:.2f}")
                else:
                    col2.metric("1년 후 통계적 예상가", "연산 불가")
                    
                high_52 = baseline['52주 최고가']
                col3.metric("52주 최고가", f"{high_52:,.0f}원" if isinstance(high_52, (int, float)) and "KS" in target_ticker else f"${high_52}" if isinstance(high_52, (int, float)) else "N/A")
                
                # LLM 프롬프트 조립
                prompt = f"""
## 실시간 금융 데이터 소스 (분석 대상: {target_ticker})
1. 기초 데이터: {baseline}
2. 몬테카를로 가치평가 모델 예측값: {mc}
3. 비정형 뉴스 텍스트 데이터: {news}

위 실시간 데이터를 바탕으로 시스템 지침(Professional Equity Research Agent 규격)에 맞게 객관적이고 밀도 높은 '종합 주식 평가 보고서'를 한국어로 작성해줘. 
국내 주식일 경우 통화 단위를 원(KRW)화 기준으로 자연스럽게 보정하여 서술하고, 수식은 LaTeX 형식으로, 실무용 엑셀 함수를 함께 포함해줘.
"""
                
                # 교수님 선택 모델 호출
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                
                # 결과 마크다운 출력
                st.markdown("---")
                st.subheader("📋 수석 애널리스트 최종 리서치 보고서")
                st.markdown(response.text)
                
                # 다운로드 버튼 기능
                st.download_button(
                    label="📥 보고서 마크다운 파일 다운로드",
                    data=response.text,
                    file_name=f"Equity_Report_{target_ticker}_{datetime.date.today()}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"분석 중 예기치 못한 오류가 발생했습니다: {e}")
