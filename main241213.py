import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd

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
</style>
""", unsafe_allow_html=True)

# OpenAIとGoogle CloudのAPIキー設定
openai.api_key = st.secrets["openai"]["api_key"]
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as cred_file:
    json.dump(dict(google_credentials_data), cred_file)
    google_credentials_path = cred_file.name
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# MP3からWAVへの変換関数
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        subprocess.run(['ffmpeg', '-y', '-i', mp3_file_path, wav_file_path], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpegの変換に失敗しました: {e}")
        return False
    return True

# 音声ファイルをチャンクに分割する関数
def generate_audio_chunks(file_path, chunk_size=4096):
    with open(file_path, 'rb') as audio_file:
        while True:
            chunk = audio_file.read(chunk_size)
            if not chunk:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

# アプリケーションUI
st.markdown("<h1 style='text-align:center;'>音声ファイル処理と話題分類</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>MP3ファイルを以下にドラッグ＆ドロップまたはクリックして選択してください。</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="mp3")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
        tmp_mp3.write(uploaded_file.getvalue())
        mp3_file_path = tmp_mp3.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    # 処理中メッセージ
    st.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # MP3→WAV変換
    progress_bar.progress(10)
    if convert_mp3_to_wav(mp3_file_path, wav_file_path):
        progress_bar.progress(40)
        try:
            with wave.open(wav_file_path, 'rb') as f:
                fr = f.getframerate()

            # 音声の文字起こし
            progress_bar.progress(50)
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=fr,
                language_code='ja-JP'
            )
            streaming_config = speech.StreamingRecognitionConfig(config=config)
            requests = generate_audio_chunks(wav_file_path)

            progress_bar.progress(70)
            responses = client.streaming_recognize(config=streaming_config, requests=requests)

            transcribed_text = ""
            for response in responses:
                for result in response.results:
                    transcribed_text += result.alternatives[0].transcript + '\n'

            # 話題分類
            progress_bar.progress(85)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "以下のテキストを重要な内容が中心となるように要約して下さい。その後、話題の項目と内容をまとめろ。項目と内容の形式は 要約: 田中さんは～ のようにしろ。"
                    },
                    {"role": "user", "content": transcribed_text}
                ]
            )
            topic_content = response['choices'][0]['message']['content'].strip()

            progress_bar.progress(100)

            # 要約部分を抽出
            lines = topic_content.split("\n")
            summary = lines[0] if lines[0].lower().startswith("要約") else "記載なし"
            topic_lines = lines[1:] if summary != "記載なし" else lines

            # 分類された話題をデータフレーム化
            categories = []
            values = []
            for line in topic_lines:
                if ":" in line:
                    c, v = line.split(":", 1)
                    categories.append(c.strip())
                    values.append(v.strip())
                else:
                    categories.append(line.strip())
                    values.append("記載なし")

            df = pd.DataFrame({"項目": categories, "内容": values})

            # 結果の表示
            st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)

            # 要約の表示
            st.markdown("### 要約")
            st.markdown(f"<div style='padding:10px; font-size:1.2em;'>{summary}</div>", unsafe_allow_html=True)

            # 分類された話題の表示
            st.markdown("### 分類された話題")

            def highlight_missing(s):
                return ['background-color: #fdd' if v == "記載なし" else '' for v in s]

            st.table(df.style.apply(highlight_missing, subset=['内容']))

        except Exception as e:
            st.error(f"処理に失敗しました: {e}")

    # 一時ファイル削除
    try:
        os.remove(mp3_file_path)
        os.remove(wav_file_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")
