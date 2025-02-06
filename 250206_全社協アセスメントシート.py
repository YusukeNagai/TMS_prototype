import os
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd
import time
import re
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# ページ設定: サイドバーをデフォルトで閉じた状態にする
st.set_page_config(
    page_title="アセスメント補助ツールver全社協アセスメントシート",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# サイドバー設定
st.sidebar.header("設定")
language_code = st.sidebar.selectbox(
    "言語コード",
    options=["ja-JP", "en-US", "es-ES", "fr-FR"],
    index=0,
    help="音声の言語コードを選択してください。"
)
chunk_length_ms = st.sidebar.slider(
    "チャンクの長さ (ミリ秒)",
    min_value=30000,
    max_value=55000,  # 最大値を55,000ミリ秒に設定
    value=50000,
    step=5000,  # ステップを5,000ミリ秒に設定
    help="音声を分割するチャンクの長さを設定します。"
)
max_workers = st.sidebar.slider(
    "並列処理のスレッド数",
    min_value=2,
    max_value=20,
    value=10,
    step=2,
    help="文字起こしの並列処理に使用するスレッド数を設定します。"
)

# CSSのスタイル設定
st.markdown("""
<style>
body {
    font-size: 1.5em;
}
h1, h2, h3 {
    font-size: 2em !important;
}
table {
    font-size: 1.3em;
}
textarea {
    font-size: 1.3em;
}
div[data-testid="stFileUploader"] {
    text-align: center;
    border: 4px dashed #ccc;
    border-radius: 15px;
    padding: 50px;
    margin: 20px auto;
    font-size: 1.5em;
    color: #333;
    background-color: #f9f9f9;
    width: 80%;
    max-width: 600px;
    cursor: pointer;
}
div[data-testid="stFileUploader"]:hover {
    background-color: #eee;
    border-color: #aaa;
}
/* デフォルトのドラッグ＆ドロップテキストを非表示に */
div[data-testid="stFileUploadDropzone"] > div:nth-child(2) {
    display: none;
}
/* 独自のテキストを表示 */
div[data-testid="stFileUploadDropzone"]::before {
    content: "MP3またはM4Aファイルをドラッグ＆ドロップするか、クリックして選択してください。";
    display: block;
    text-align: center;
    font-size: 1.5em;
    color: #333;
    padding: 20px;
}
.audio-player {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# OpenAIとGoogle CloudのAPIキー設定
openai.api_key = st.secrets["openai"]["api_key"]
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as cred_file:
    json.dump(dict(google_credentials_data), cred_file)
    google_credentials_path = cred_file.name
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# オーディオファイルからWAVへの変換関数
def convert_to_wav(input_audio_segment, output_file_path):
    try:
        # pydubを使用してAudioSegmentから直接WAVにエクスポート
        input_audio_segment.export(output_file_path, format="wav", bitrate="16k", parameters=["-ar", "16000", "-ac", "1"])
    except Exception as e:
        st.error(f"WAVへの変換に失敗しました: {e}")
        return False
    return True

# WAVファイルを分割する関数（メモリ内で処理）
def split_audio_to_chunks(wav_file_path, chunk_length_ms=50000):
    audio = AudioSegment.from_wav(wav_file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunks.append(chunk)
    return chunks

# Google Speech-to-Textで文字起こしを行う関数（各チャンク用）
def transcribe_chunk(chunk_audio, client, language_code='ja-JP'):
    try:
        with BytesIO() as wav_buffer:
            chunk_audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            content = wav_buffer.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code
        )
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        return transcript
    except Exception as e:
        return f"文字起こしに失敗しました: {e}"

st.markdown("<h1 style='text-align:center;'>アセスメント補助ツール(全社協アセスメントシート)</h1>", unsafe_allow_html=True)

# ファイルアップロード部分を日本語化（複数ファイル対応に変更）
uploaded_files = st.file_uploader(
    "MP3またはM4Aファイルをドラッグ＆ドロップするか、クリックして選択してください。",
    type=["mp3", "m4a"],
    accept_multiple_files=True  # 複数ファイルを受け付けるように変更
)

if uploaded_files:
    # 複数ファイルがアップロードされた場合の処理
    if len(uploaded_files) > 1:
        st.markdown("### アップロードされた音声ファイルの結合")
        combined = AudioSegment.empty()
        audio_players = []
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                audio = AudioSegment.from_file(uploaded_file)
                combined += audio
                # 各ファイルの再生
                audio_bytes = uploaded_file.read()
                audio_players.append((uploaded_file.name, audio_bytes))
                st.audio(audio_bytes, format=uploaded_file.type)
            except Exception as e:
                st.error(f"{uploaded_file.name} の読み込みに失敗しました: {e}")
        # 結合した音声の再生
        st.markdown("### 結合後の音声ファイル")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_combined_wav:
            combined.export(tmp_combined_wav.name, format="wav")
            combined_wav_path = tmp_combined_wav.name
            combined_wav_bytes = open(combined_wav_path, "rb").read()
            st.audio(combined_wav_bytes, format="audio/wav")
    else:
        # 単一ファイルの場合の処理
        uploaded_file = uploaded_files[0]
        input_bytes = uploaded_file.read()
        st.markdown("### アップロードした音声ファイル")
        st.audio(input_bytes, format=uploaded_file.type)
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            combined_wav_path = tmp_wav.name
        # pydubを使用して単一ファイルをAudioSegmentに変換
        try:
            audio = AudioSegment.from_file(BytesIO(input_bytes))
            audio.export(combined_wav_path, format="wav", bitrate="16k", parameters=["-ar", "16000", "-ac", "1"])
        except Exception as e:
            st.error(f"WAVへの変換に失敗しました: {e}")
            st.stop()

    if len(uploaded_files) > 1:
        # 複数ファイルを結合した場合のWAVファイルのパスはcombined_wav_path
        wav_file_path = combined_wav_path
    else:
        # 単一ファイルの場合のWAVファイルのパスもcombined_wav_path
        wav_file_path = combined_wav_path

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav_final:
        wav_final_path = tmp_wav_final.name

    # 処理中メッセージと進捗バー
    st.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 処理時間を記録する辞書
    timing = {}

    try:
        # 入力ファイルからWAVへの変換
        start_time = time.perf_counter()
        status_text.text("ステップ 1/5: WAV形式への変換中...")
        progress_bar.progress(10)
        if convert_to_wav(AudioSegment.from_file(wav_file_path), wav_final_path):
            end_time = time.perf_counter()
            timing['ファイル変換 (WAV)'] = end_time - start_time
            progress_bar.progress(20)
            try:
                # 音声分割
                start_time = time.perf_counter()
                status_text.text("ステップ 2/5: 音声を分割中...")
                chunks = split_audio_to_chunks(wav_final_path, chunk_length_ms=chunk_length_ms)  # チャンク長をサイドバーから取得
                end_time = time.perf_counter()
                timing['音声分割'] = end_time - start_time
                progress_bar.progress(30)

                # Google Speech-to-Text クライアントの再利用
                client = speech.SpeechClient()

                # 並列処理で文字起こし
                start_time = time.perf_counter()
                status_text.text("ステップ 3/5: 文字起こしを開始...")
                transcripts = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:  # スレッド数をサイドバーから取得
                    futures = [executor.submit(transcribe_chunk, chunk, client, language_code) for chunk in chunks]
                    for i, future in enumerate(as_completed(futures), start=1):
                        result = future.result()
                        transcripts.append(result)
                        # 進捗を更新 (最大30→80程度に分配)
                        current_progress = 30 + int((i / len(chunks)) * 50)
                        progress_bar.progress(min(current_progress, 80))
                full_transcript = "\n".join(transcripts)
                end_time = time.perf_counter()
                timing['音声の文字起こし(並列)'] = end_time - start_time

                # 音声の再生（変換後のWAVファイル）
                st.markdown("### 変換後のWAVファイル")
                with open(wav_final_path, "rb") as wav_file:
                    wav_bytes = wav_file.read()
                    st.audio(wav_bytes, format="audio/wav")

                progress_bar.progress(85)
                status_text.text("ステップ 4/5: 話題分類を実行中...")

                # OpenAI APIで沖縄式アセスメントシートに基づく話題分類・要約・整理を実施
                start_time = time.perf_counter()
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # 正しいモデル名に修正
                    messages=[
                        {
                            "role": "system", 
                            "content": "あなたは介護領域における幅広い専門知識を持つアシスタントです。特にケアマネジャー向けの情報に関して専門的な回答を提供できます。情報がない場合は必ず「記載なし」と記してください。"
                        },
                        {
                            "role": "user",
                            "content": f"""
最下部に述べる音声記録を参考に、以下の手順でテキストの要約と内容整理を行ってください。

1. **要約作成**
   - 音声記録全体の中から重要な内容を抽出し、簡潔に要約してください。
   - 情報が不足している場合は「記載なし」と記してください。

2. **沖縄式アセスメントシートの項目ごとに整理**
   以下の各項目について、音声記録の内容を漏れなく整理し、詳細に記述してください。
   - 各項目で、会話が全くなかった場合は「特記なし」と記載してください。
   - ケアマネジャーが質問したが回答がなかった場合は「空白」とし、理由（例：「○○という質問に回答無し」）を記述してください。
   - 最後に、ヒアリングが不足している重要な項目について、次回のヒアリングで確認すべき点を提案してください。

【沖縄式アセスメントシートの記入項目】
1. 【フェースシート（基本情報）】
   - 訪問・電話・来所の区分
   - 介護保険負担割合
   - アセスメント実施日
   - 相談経路（紹介者）
   - 本人氏名
   - 性別
   - 生年月日
   - 住所
   - 電話番号
   - 要介護認定
   - 認知症レベル
   - 緊急連絡先（例：長女など）
2. 【家族状況とインフォーマルな支援の状況】
   - 家族構成（各家族の状況と支援内容）
   - 地域での支援（例：近所の民生委員、自治会、ボランティア団体）
3. 【サービス利用状況】
   - 訪問介護（サービス内容・頻度）
   - デイサービス（サービス内容・頻度）
   - 訪問看護
   - 福祉用具貸与
   - 配食サービス
4. 【住居等の状況】
   - 居住環境（住宅の種類、階数、設備等）
   - バリアフリー状況（トイレ、浴室、居室、玄関の状況）
   - 使用中の福祉用具
5. 【本人の健康状態・受診状況】
   - 既往歴
   - 現在の診断名
   - 服薬状況
   - 通院状況（内科、整形外科、歯科など）
6. 【本人の基本動作と援助内容】
   - 寝返り
   - 起き上がり
   - 立ち上がり
   - 歩行
   - 入浴・洗身
   - 食事摂取
   - 排泄
7. 【認知機能】
   - 記憶力低下
   - 見当識障害
   - 認知症レベル
8. 【精神・行動障害】
   - 被害妄想
   - 昼夜逆転傾向
   - 社会的行動の変化
9. 【医療・健康管理】
   - 糖尿病管理
   - リハビリの必要性
   - 在宅医療の有無
10. 【社会生活】
    - 金銭管理
    - 買い物
    - 趣味活動
11. 【総合評価とケアプランの方向性】
    - 現状の課題
    - 今後のケアプランの方向性

# 出力形式

- 各項目の内容は、詳細かつ丁寧な文でまとめてください。
- 出力例:
  要約: [音声記録の要約]
  1. フェースシート（基本情報）
     訪問・電話・来所の区分: [内容]
     介護保険負担割合: [内容]
     ・・・
  11. 総合評価とケアプランの方向性
     現状の課題: [内容]
     今後のケアプランの方向性: [内容]
- 最後に、ヒアリングが不足している重要な項目について、次回のヒアリングで確認すべき点を提案してください。

以下に、音声記録のテキストを示します。
"""
                        },
                        {"role": "user", "content": full_transcript}
                    ]
                )
                topic_content = response['choices'][0]['message']['content'].strip()
                end_time = time.perf_counter()
                timing['話題分類 (OpenAI GPT-4o)'] = end_time - start_time

                progress_bar.progress(100)
                status_text.text("ステップ 5/5: 処理完了しました。")

                # 結果の表示
                st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)

                # GPTの出力を常に表示
                st.markdown("### GPTの出力")
                st.markdown(f"<div style='padding:10px; font-size:1.2em; white-space: pre-wrap;'>{topic_content}</div>", unsafe_allow_html=True)

                # 結果のダウンロード
                st.markdown("### 結果のダウンロード")
                topic_bytes = topic_content.encode('utf-8')
                st.download_button(
                    label="GPTの出力をダウンロード",
                    data=topic_bytes,
                    file_name="topic_content.txt",
                    mime="text/plain"
                )

                # 処理時間の表示を折りたたみ可能なセクションに変更
                with st.expander("処理時間を表示"):
                    st.markdown("### 処理時間")
                    timing_df = pd.DataFrame({
                        "ステップ": list(timing.keys()),
                        "所要時間 (秒)": [f"{v:.2f}" for v in timing.values()]
                    })
                    st.table(timing_df)

                    # 処理時間のダウンロード
                    timing_csv = timing_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="処理時間をCSVでダウンロード",
                        data=timing_csv,
                        file_name="processing_time.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"処理に失敗しました: {e}")

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")

    # 一時ファイル削除
    try:
        os.remove(wav_final_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")

    # リセットボタン
    if st.button("新しいファイルをアップロードする"):
        st.experimental_rerun()
