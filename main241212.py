import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd

# 全体的な文字サイズを更に大きくするCSS
st.markdown("""
<style>
body {
    font-size: 1.5em;  /* 全体をさらに大きく */
}
h1, h2, h3 {
    font-size: 2em !important;  /* 見出しもさらに大きく */
}
table {
    font-size: 1.3em;
}
textarea {
    font-size: 1.3em;
}
/* アップロードエリアやテキストを目立たせる */
#upload-area {
    text-align:center;
    font-size: 2.5em;
    font-weight: bold;
    margin: 20px auto;
    border: 4px dashed #ccc;
    border-radius: 15px;
    padding: 50px;
    width: 80%;
    max-width: 600px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# OpenAI APIキーをStreamlit Secretsから取得
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit SecretsからGoogle Cloud認証情報を取得
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]

# JSONファイルとして書き出し
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as cred_file:
    json.dump(dict(google_credentials_data), cred_file)
    google_credentials_path = cred_file.name

# 環境変数を設定
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

# ================================
# UI 改善
# ================================
st.markdown("<h1 style='text-align:center; font-size: 3em;'>音声ファイルの処理と話題分類</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.5em;'>MP3ファイルを下のエリアにドラッグ＆ドロップまたはクリックして選択してください。</p>", unsafe_allow_html=True)

# ドラッグ＆ドロップエリアを大きく目立たせる
st.markdown("<div id='upload-area'>ここにファイルをドラッグ＆ドロップ！</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="mp3")

if uploaded_file is not None:
    # ファイルアップロード後すぐに処理開始
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
        tmp_mp3.write(uploaded_file.getvalue())
        mp3_file_path = tmp_mp3.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    # 処理中表示
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
                            "content": "以下のテキストをケアマネジャーさんにとって必要な内容が中心となるように要約して下さい。その後、下に指定する項目ごとに分類してください。項目と内容の形式は 名前: 田中太郎 のようにしろ。ただし音声記録から埋めることができない項目は”記載なし”と記せ。項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, 投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, 記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, 息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, 現在の訪問介護"
                        },
                        {"role": "user", "content": transcribed_text}
                    ]
                )
                topic_content = response['choices'][0]['message']['content'].strip()

                progress_bar.progress(100)
                processing_placeholder.empty()

                # 結果表示
                st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)

                # 文字起こし結果表示
                st.markdown("### 文字起こし結果")
                st.text_area("Transcribed Text", transcribed_text, height=200)

                # 全角コロンを半角コロンに統一してから分割
                lines = [line for line in topic_content.split('\n') if line.strip()]

                # 要約行と項目行分けて処理
                merged_lines = []
                i = 0
                while i < len(lines):
                    current_line = lines[i].replace('：', ':')
                    if ':' in current_line:
                        cat, val = current_line.split(':', 1)
                        cat = cat.strip()
                        val = val.strip()
                        if val == '' and i + 1 < len(lines):
                            # 次の行を内容として結合
                            next_line = lines[i+1].strip()
                            val = next_line
                            i += 1
                        merged_lines.append(f"{cat}: {val}")
                    else:
                        # コロンがない行はそのまま
                        merged_lines.append(current_line)
                    i += 1

                # 要約を抜き出す
                summary_text = ""
                final_lines = []
                for ml in merged_lines:
                    if ml.lower().startswith("要約:"):
                        # 要約行の場合
                        _, s_val = ml.split(':', 1)
                        summary_text = s_val.strip()
                    else:
                        final_lines.append(ml)

                # 要約表示
                if summary_text:
                    st.markdown("### 要約")
                    st.markdown(f"<div style='border:1px solid #ccc; padding:10px; font-size:1.2em;'>{summary_text}</div>", unsafe_allow_html=True)

                # テーブル化
                categories = []
                values = []
                for line in final_lines:
                    if ':' in line:
                        c, v = line.split(':', 1)
                        categories.append(c.strip())
                        values.append(v.strip())
                    else:
                        categories.append(line.strip())
                        values.append("記載なし")

                # 分類された話題テーブル表示
                st.markdown("### 分類された話題")
                df = pd.DataFrame({"項目": categories, "内容": values})
                
                # "記載なし"を目立たせるスタイル
                def highlight_missing(s):
                    return ['background-color: #fdd' if v == "記載なし" else '' for v in s]

                st.table(df.style.apply(highlight_missing, subset=['内容']))

                st.markdown("<p style='text-align:center;'>以上の結果を参考にしてください。</p>", unsafe_allow_html=True)

            except Exception as e:
                processing_placeholder.empty()
                st.error(f"話題分類に失敗しました: {e}")
        except wave.Error as e:
            processing_placeholder.empty()
            st.error(f"WAVファイルの読み込みに失敗しました: {e}")
    else:
        processing_placeholder.empty()
        st.error("MP3からWAVへの変換が失敗しました。")

    # 一時ファイルのクリーンアップ
    try:
        os.remove(mp3_file_path)
        os.remove(wav_file_path)
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")
