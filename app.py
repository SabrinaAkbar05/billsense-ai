"""Premium Streamlit frontend for BillSense AI."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from extract import extract_bill
from answer_questions import answer_question

st.set_page_config(page_title='BillSense AI | Utility Bill Intelligence', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background: #0b1020; color:#eef2ff;}
[data-testid="stSidebar"] {background: #11182e;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stMetric"] {background: #18223b; padding: 17px; border: 1px solid #303a56; border-radius: 15px;}
[data-testid="stMetricLabel"] {color:#aab7d6;}
div.stButton > button[kind="primary"] {background: #7c62e8; border-color:#7c62e8; color:white; border-radius: 12px;}
[data-testid="stFileUploader"] {border: 1px dashed #7775b8; border-radius: 16px; padding: 8px;}
</style>''', unsafe_allow_html=True)

st.sidebar.markdown('## ⚡ BillSense AI')
st.sidebar.caption('Utility Bill Intelligence · Competition Edition')
page = st.sidebar.radio('Navigate', ['Analyze a bill','Ask AI','Batch processing','About'], label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('Powered by Gemini Vision · KE / LESCO / IESCO')

if 'bill' not in st.session_state: st.session_state.bill = None
if 'image_bytes' not in st.session_state: st.session_state.image_bytes = None
if 'image_suffix' not in st.session_state: st.session_state.image_suffix = '.png'
if 'messages' not in st.session_state: st.session_state.messages = []

def image_path():
    tmp = tempfile.NamedTemporaryFile(suffix=st.session_state.image_suffix, delete=False)
    tmp.write(st.session_state.image_bytes)
    tmp.close()
    return Path(tmp.name)

def fmt(v, prefix=''):
    return 'Not available' if v is None else f'{prefix}{v:,.2f}' if isinstance(v,(float,int)) else str(v)

if page == 'Analyze a bill':
    st.title('⚡ Understand your electricity bill')
    st.caption('Upload a bill to turn complicated charges into clear, grounded insights.')
    left, right = st.columns([1.1, 1], gap='large')
    with left:
        uploaded = st.file_uploader('Upload electricity bill', type=['png','jpg','jpeg'])
        if uploaded:
            st.image(uploaded, caption='Original bill', use_container_width=True)
            if st.button('✨ Analyze Bill', type='primary', use_container_width=True):
                content = uploaded.getvalue()
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix.lower(), delete=False) as tmp:
                    tmp.write(content)
                    path = Path(tmp.name)
                try:
                    with st.spinner('Reading the original bill with Gemini Vision...'):
                        data = extract_bill(path)
                    st.session_state.bill = data
                    st.session_state.image_bytes = content
                    st.session_state.image_suffix = path.suffix
                    st.session_state.messages = []
                    st.success('Bill analyzed!')
                except Exception as exc:
                    st.error(f'Extraction failed: {exc}')
                finally:
                    path.unlink(missing_ok=True)
    with right:
        bill = st.session_state.bill
        if not bill:
            st.info('Your bill summary will appear here after analysis.')
        else:
            st.subheader('Bill overview')
            st.caption(f"{bill.get('provider') or 'Unknown provider'} · {bill.get('bill_month') or 'Unknown month'} · {bill.get('tariff') or 'Tariff unavailable'}")
            c1,c2 = st.columns(2)
            c1.metric('Current bill', fmt(bill.get('current_bill'),'Rs. '))
            c2.metric('Units consumed', fmt(bill.get('units_consumed')))
            c3,c4 = st.columns(2)
            c3.metric('Taxes', fmt(bill.get('total_taxes'),'Rs. '))
            c4.metric('Payable within due date', fmt(bill.get('payable_within_due_date'),'Rs. '))
            st.caption(f"Due date: {bill.get('due_date') or 'Not available'} · Negative payable means credit when printed as CR.")
            charges = [{'Category':str(x.get('type','other')).replace('_',' ').title(),'Amount':x.get('amount',0)} for x in bill.get('charges',[]) if isinstance(x,dict) and isinstance(x.get('amount'),(int,float))]
            taxes = [{'Category':str(x.get('type','other')).replace('_',' ').title(),'Amount':x.get('amount',0)} for x in bill.get('taxes',[]) if isinstance(x,dict) and isinstance(x.get('amount'),(int,float))]
            entries = charges+taxes
            if entries:
                st.subheader('Charges & taxes')
                frame = pd.DataFrame(entries).groupby('Category',as_index=False)['Amount'].sum()
                st.bar_chart(frame.set_index('Category'))
                with st.expander('Itemized breakdown'):
                    st.dataframe(frame, hide_index=True, use_container_width=True)
            with st.expander('Structured Level 1 JSON'):
                st.json(bill)
            st.download_button('⬇ Download extracted JSON', json.dumps(bill,ensure_ascii=False,indent=2), file_name='bill_extraction.json', mime='application/json', use_container_width=True)
            if st.button('💬 Ask a question about this bill', use_container_width=True):
                st.session_state.nav_hint = True
                st.info('Select “Ask AI” from the sidebar.')

elif page == 'Ask AI':
    st.title('💬 Ask BillSense AI')
    st.caption('Answers are grounded in your original bill image and its extracted data.')
    if st.session_state.bill is None or st.session_state.image_bytes is None:
        st.warning('Analyze a bill first using the “Analyze a bill” page.')
    else:
        for message in st.session_state.messages:
            with st.chat_message(message['role']): st.markdown(message['content'])
        question = st.chat_input('Example: What percentage of my bill is taxes?')
        if question:
            st.session_state.messages.append({'role':'user','content':question})
            with st.chat_message('user'): st.markdown(question)
            with st.chat_message('assistant'):
                path = image_path()
                try:
                    with st.spinner('Inspecting the bill...'):
                        reply = answer_question(path, st.session_state.bill, question)
                    st.markdown(reply)
                    st.session_state.messages.append({'role':'assistant','content':reply})
                except Exception as exc:
                    st.error(f'Could not answer: {exc}')
                finally:
                    path.unlink(missing_ok=True)

elif page == 'Batch processing':
    st.title('📦 Competition batch processing')
    st.markdown('Run the official test pipeline from the terminal. It preserves CSV template row order, retries transient failures, and saves checkpoints.')
    st.code('python main.py --train-check\npython main.py\npython package_submission.py',language='powershell')
    st.info('The test data must exist in test/bills and test/csv. Batch submission runs locally; this dashboard does not upload your test data.')
    for filename in ('output/level1.csv','output/level2.csv'):
        p = Path(filename)
        if p.exists():
            st.success(f'{filename} is available')
            st.download_button(f'Download {p.name}',p.read_bytes(),file_name=p.name,mime='text/csv',key=filename)
        else: st.caption(f'{filename} has not been generated yet.')

else:
    st.title('About BillSense AI')
    st.write('BillSense AI transforms electricity bill images into structured data and evidence-grounded explanations.')
    st.markdown('**Supported providers:** KE / K-Electric, LESCO, IESCO. **Model:** Gemini 3.1 Flash-Lite (configurable via `.env`).')
    st.warning('AI may make mistakes. Verify readings, bill credits, arithmetic, and any ambiguous net-metering fields against the original image.')
