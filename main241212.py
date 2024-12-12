import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd

# CSSでfile_uploaderのスタイルを調整
st.markdown("""
<style>
/* file_uploaderの全体を大きくカスタマイズ */
div[data-testid="stFileUploader"] {
    text-align: center;
    border: 4px dashed #ccc; /* 点線の枠 */
    border-radius: 15px;    /* 角を丸く */
    padding: 50px;          /* 内側の余白を広げる */
    margin: 20px auto;      /* 周囲の余白を中央揃え */
    font-size: 1.5em;       /* 文字サイズを大きく */
    color: #333;            /* テキストの色 */
    background-color: #f9f9f9; /* 背景色 */
    width: 80%;             /* 全体幅 */
    max-width: 600px;       /* 最大幅を設定 */
    cursor: pointer;        /* ホバーで手のアイコンに */
}

/* file_uploaderがhoverされた時の効果 */
div[data-testid="stFileUploader"]:hover {
    background-color: #eee; /* 背景色を少し変更 */
    border-color: #aaa;     /* 枠線の色を濃く */
}
</style>
""", unsafe_allow_html=True)

# OpenAI APIキーの設定
openai.api_key = st.secrets["openai"]["api_key"]

# Google Cloud Credentials
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as cred_file:
    json.dump(dict(google_credentials_data), cred_file)
    google_credentials_path = cred_file.name
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# MP3ファイルをWAVファイルに変換する関数
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

# StreamlitアプリケーションのUI設定
st.markdown("<h1 style='text-align:center;'>音声ファイル処理と話題分類</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>MP3ファイルを下のエリアにドラッグ＆ドロップまたはクリックして選択してください。</p>", unsafe_allow_html=True)

# ファイルアップロード
tempfile.tempdir = '/tmp'  # 一時ファイルの保存先
uploaded_file = st.file_uploader("ここにファイルをドラッグ＆ドロップ！", type="mp3")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
        tmp_mp3.write(uploaded_file.getvalue())
        mp3_file_path = tmp_mp3.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    processing_placeholder = st.empty()
    processing_placeholder.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>しばらくお待ちください</p>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # (1) MP3→WAV変換
    progress_bar.progress(10)
    if convert_mp3_to_wav(mp3_file_path, wav_file_path):
        progress_bar.progress(40)
        try:
            with wave.open(wav_file_path, 'rb') as f:
                fr = f.getframerate()

            # (2) 文字起こし
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

            # (3) 話題分類
            progress_bar.progress(85)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system", 
                            "content": "以下のテキストをケアマネジャーさんにとって必要な内容が中心となるように要約して下さい。その後、下に指定する項目ごとに分類してください。項目と内容の形式は 名前: 田中太郎 のようにしろ。ただし音声記録から埋めることができない項目は\"記載なし\"と記せ。項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, 投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, 記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, 息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, 現在の訪問介護"
                        },
                        {"role": "user", "content": transcribed_text}
                    ]
                )
                topic_content = response['choices'][0]['message']['content'].strip()

                progress_bar.progress(100)
                processing_placeholder.empty()

                st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)

                # 文字起こし結果表示
                st.markdown("### 文字起こし結果")
                st.text_area("Transcribed Text", transcribed_text, height=200)

                # 分類された話題をテーブル表示
                lines = [line for line in topic_content.split('\n') if line.strip()]
                categories = []
                values = []
                for line in lines:
                    if ':' in line:
                        c, v = line.split(':', 1)
                        categories.append(c.strip())
                        values.append(v.strip())
                    else:
                        categories.append(line.strip())
                        values.append("記載なし")

                st.markdown("### 分類された話題")
                df = pd.DataFrame({"項目": categories, "内容": values})
                st.table(df)

            except Exception as e:
                processing_placeholder.empty()
                st.error(f"話題分類に失敗しました: {e}")
        except wave.Error as e:
            processing_placeholder.empty()
            st.error(f"WAVファイルの読み込みに失敗しました: {e}")
    else:
        processing_placeholder.empty()
        st.error("MP3からWAVへの変換が失敗しました。")

    # 一時ファイル削除
    try:
        os.remove(mp3_file_path)
        os.remove(wav_file_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")
