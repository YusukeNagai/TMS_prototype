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
    page_title="アセスメント補助ツールver3.0沖縄式アセスメントシート",
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
    max_value=55000,
    value=50000,
    step=5000,
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
div[data-testid="stFileUploadDropzone"] > div:nth-child(2) {
    display: none;
}
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

st.markdown("<h1 style='text-align:center;'>アセスメント補助ツール</h1>", unsafe_allow_html=True)

# ファイルアップロード（複数ファイル対応）
uploaded_files = st.file_uploader(
    "MP3またはM4Aファイルをドラッグ＆ドロップするか、クリックして選択してください。",
    type=["mp3", "m4a"],
    accept_multiple_files=True
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
                audio_bytes = uploaded_file.read()
                audio_players.append((uploaded_file.name, audio_bytes))
                st.audio(audio_bytes, format=uploaded_file.type)
            except Exception as e:
                st.error(f"{uploaded_file.name} の読み込みに失敗しました: {e}")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            combined_wav_path = tmp_wav.name
        try:
            audio = AudioSegment.from_file(BytesIO(input_bytes))
            audio.export(combined_wav_path, format="wav", bitrate="16k", parameters=["-ar", "16000", "-ac", "1"])
        except Exception as e:
            st.error(f"WAVへの変換に失敗しました: {e}")
            st.stop()

    # 単一・複数いずれの場合もWAVファイルのパスはcombined_wav_path
    wav_file_path = combined_wav_path

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav_final:
        wav_final_path = tmp_wav_final.name

    st.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    timing = {}

    try:
        # ステップ1: 入力ファイルのWAV変換
        start_time = time.perf_counter()
        status_text.text("ステップ 1/5: WAV形式への変換中...")
        progress_bar.progress(10)
        if convert_to_wav(AudioSegment.from_file(wav_file_path), wav_final_path):
            end_time = time.perf_counter()
            timing['ファイル変換 (WAV)'] = end_time - start_time
            progress_bar.progress(20)
            try:
                # ステップ2: 音声の分割
                start_time = time.perf_counter()
                status_text.text("ステップ 2/5: 音声を分割中...")
                chunks = split_audio_to_chunks(wav_final_path, chunk_length_ms=chunk_length_ms)
                end_time = time.perf_counter()
                timing['音声分割'] = end_time - start_time
                progress_bar.progress(30)

                # ステップ3: Google Speech-to-Textによる文字起こし
                client = speech.SpeechClient()
                start_time = time.perf_counter()
                status_text.text("ステップ 3/5: 文字起こしを開始...")
                transcripts = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(transcribe_chunk, chunk, client, language_code) for chunk in chunks]
                    for i, future in enumerate(as_completed(futures), start=1):
                        result = future.result()
                        transcripts.append(result)
                        current_progress = 30 + int((i / len(chunks)) * 50)
                        progress_bar.progress(min(current_progress, 80))
                full_transcript = "\n".join(transcripts)
                end_time = time.perf_counter()
                timing['音声の文字起こし(並列)'] = end_time - start_time

                # 変換後のWAVファイル再生
                st.markdown("### 変換後のWAVファイル")
                with open(wav_final_path, "rb") as wav_file:
                    wav_bytes = wav_file.read()
                    st.audio(wav_bytes, format="audio/wav")

                progress_bar.progress(85)
                status_text.text("ステップ 4/5: 話題分類を実行中...")

                # ステップ4: OpenAI APIを用いて沖縄式アセスメントシート形式の話題分類・要約・整理
                start_time = time.perf_counter()
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # 使用するモデル名を適宜修正してください
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたは介護領域における専門的な知識を持つアシスタントです。利用者情報が不足している場合は必ず「記載なし」または「特記なし」と記入してください。"
                        },
                        {
                            "role": "user",
                            "content": f"""
以下の手順に従い、音声記録のテキストから要約および内容整理を実施してください。

1. **要約作成**  
   - 音声記録全体から重要な内容を抽出し、簡潔に要約してください。  
   - 情報が不足している場合は「記載なし」と記入してください。

2. **沖縄式アセスメントシートの各項目ごとに整理**  
   以下の各項目について、音声記録の内容を漏れなく整理し、詳細に記述してください。  
   - 各項目で会話が全くなかった場合は「特記なし」と記入してください。  
   - ケアマネジャーが質問したが回答がなかった場合は「空白」とし、理由（例：「○○という質問に回答無し」）を記述してください。  
   - 最後に、ヒアリングが不足している重要な項目について、次回のヒアリングで確認すべき点を提案してください。

【沖縄式アセスメントシートの記入項目】
1. 【基本情報】  
   - 氏名、性別、生年月日、住所、連絡先、家族構成、緊急連絡先
2. 【健康状態】  
   - 現在の健康状態（服薬や受診に関する状況・自身の健康に対する理解や意識の状況）  
   - 既往歴、持病  
   - 服薬状況  
   - 受診状況
3. 【心身機能・身体構造】  
   - 移動方法（室内および屋外の状況を詳しく記載）  
   - 階段昇降の可否（室内・屋外の状況の詳細）  
   - 交通機関の利用・車の運転等に関する状況  
   - コミュニケーション（電話は統合し、コミュニケーション機器・方法等の記載）
4. 【活動】  
   - 日常生活の活動レベル、ADL/IADL評価等
5. 【生活状況】  
   - 1日及び1週間の過ごし方  
   - 家族等の状況や関わり（家族関係等で特記すべき事項、家庭での役割も統合）
6. 【その他】  
   - 寝具や衣類の管理（Eその他として追加）  
   - 審査会意見
7. 【環境因子】  
   - 促進因子・阻害因子の具体例

【詳細な記入例】
1. 基本情報  
   氏名：山田 太郎  
   性別：男性  
   生年月日：1945年5月10日（78歳）  
   住所：沖縄県那覇市○○町1-2-3  
   連絡先：098-XXX-XXXX  
   家族構成：妻（74歳）、長男（45歳、県外在住）、長女（42歳、県外在住）  
   緊急連絡先：長男 山田 健一（携帯：090-XXX-XXXX）
2. 健康状態  
   現在の健康状態：高血圧、糖尿病（インスリン治療中）、脊柱管狭窄症による歩行障害あり  
   既往歴：2018年の脳梗塞、2020年の腰椎圧迫骨折  
   服薬状況：高血圧薬、糖尿病薬、鎮痛剤など  
   受診状況：内科・整形外科の定期受診
3. 心身機能・身体構造  
   移動方法：室内は杖歩行、屋外は車椅子利用（詳細な状況を記載）  
   階段昇降：不可（理由を明記）  
   交通機関・運転：運転不可、またはバス利用の状況  
   コミュニケーション：意思疎通可能（電話は他の手段と統合し、使用している機器や方法を記載）
4. 活動  
   日常生活の活動レベル：例）一部介助  
   ADL/IADL評価：寝返り、起き上がり、歩行など
5. 生活状況  
   1日及び1週間の過ごし方：例）テレビ視聴、昼寝が多い等  
   家族等の状況や関わり：家族構成、家庭内での役割、支援状況等
6. その他  
   寝具や衣類の管理：例）妻が対応（Eその他として記載）  
   審査会意見：介護サービスの適正利用など
7. 環境因子  
   促進因子・阻害因子：住環境のバリアフリー状況、地域の支援体制など

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

                # 処理時間の表示（展開可能なセクション）
                with st.expander("処理時間を表示"):
                    st.markdown("### 処理時間")
                    timing_df = pd.DataFrame({
                        "ステップ": list(timing.keys()),
                        "所要時間 (秒)": [f"{v:.2f}" for v in timing.values()]
                    })
                    st.table(timing_df)
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

    # 一時ファイルの削除
    try:
        os.remove(wav_final_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")

    # リセットボタン
    if st.button("新しいファイルをアップロードする"):
        st.experimental_rerun()
