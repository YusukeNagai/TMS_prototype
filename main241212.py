import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd

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
# 全画面簡易操作モードかつステップバイステップを採用。
# ・最初の画面では大きなアップロードエリアのみ
# ・アップロードされたら直ちに処理開始（次へボタンは不要）
# ・処理中は「処理中」「変換中」「文字起こし中」など明確な表示
# ・結果表示はテーブルで分類項目ごとに見やすく表示
# ・アイコンやシンプルなテキストで操作を誘導

st.markdown("<h1 style='text-align:center; font-size: 2.5em;'>音声ファイルの処理と話題分類</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2em;'>MP3ファイルをドラッグ＆ドロップまたはボタンをクリックしてアップロードしてください。</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:3em;'>➕</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="mp3")

if uploaded_file is not None:
    # ファイルアップロード後はすぐに処理を開始する
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
        tmp_mp3.write(uploaded_file.getvalue())
        mp3_file_path = tmp_mp3.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    # 処理中の画面表示
    processing_placeholder = st.empty()
    processing_placeholder.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>しばらくお待ちください</p>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # 進捗1: MP3→WAV変換
    progress_bar.progress(10)
    if convert_mp3_to_wav(mp3_file_path, wav_file_path):
        progress_bar.progress(40)
        try:
            with wave.open(wav_file_path, 'rb') as f:
                fr = f.getframerate()

            # 進捗2: 文字起こし設定
            progress_bar.progress(50)
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=fr,
                language_code='ja-JP'
            )
            streaming_config = speech.StreamingRecognitionConfig(config=config)
            requests = generate_audio_chunks(wav_file_path)

            # 進捗3: 文字起こし中
            progress_bar.progress(70)
            responses = client.streaming_recognize(config=streaming_config, requests=requests)
            transcribed_text = ""
            for response in responses:
                for result in response.results:
                    transcribed_text += result.alternatives[0].transcript + '\n'

            # 進捗4: 話題分類中
            progress_bar.progress(85)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system", 
                            "content": "以下のテキストをケアマネジャーさんにとって必要な内容が中心となるように要約して下さい。その後、下に指定する項目ごとに分類してください。ただし音声記録から埋めることができない項目は”記載なし”と記せ。項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, 投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, 記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, 息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, 現在の訪問介護"
                        },
                        {"role": "user", "content": transcribed_text}
                    ]
                )
                topic_content = response['choices'][0]['message']['content'].strip()
                
                # 進捗5: 結果表示準備
                progress_bar.progress(100)
                processing_placeholder.empty()

                # 結果表示画面
                st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)
                st.markdown("### 文字起こし結果")
                st.text_area("Transcribed Text", transcribed_text, height=200)

                st.markdown("### 分類された話題")
                # 結果を表形式で表示
                # 項目ごとに':'区切りで扱える場合はここで処理する
                # そうでなければ単純に行で表示
                
                # ここでは「項目:内容」形式になっていると想定し、DataFrame化を試みる
                lines = [line for line in topic_content.split('\n') if line.strip() != '']
                categories = []
                values = []
                for line in lines:
                # 「項目: 内容」の形式を想定（半角コロンを使用）
                    if ':' in line:
                        cat, val = line.split(':', 1)
                        categories.append(cat.strip())
                        values.append(val.strip())
                    else:
                        # コロンが含まれない場合は記載なし扱い
                        categories.append(line.strip())
                        values.append("記載なし")

                df = pd.DataFrame({"項目": categories, "内容": values})


                df = pd.DataFrame({"項目": categories, "内容": values})
                # "記載なし"を目立たせるため、スタイルを付与（背景色や文字色など）
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
