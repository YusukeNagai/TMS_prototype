import os
import wave
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

# オーディオファイルからWAVへの変換関数
def convert_to_wav(input_file_path, output_file_path):
    try:
        # ffmpegを使用して入力ファイルをWAVに変換
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file_path,
            '-ar', '16000', '-ac', '1', output_file_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpegの変換に失敗しました: {e}")
        return False
    return True

# WAVファイルを分割する関数
def split_audio_to_chunks(wav_file_path, chunk_length_ms=50000):
    audio = AudioSegment.from_wav(wav_file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        # 一時ファイルとして保存
        tmp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        chunk.export(tmp_chunk.name, format="wav")
        chunks.append(tmp_chunk.name)
    return chunks

# Google Speech-to-Textで文字起こしを行う関数（各チャンク用）
def transcribe_chunk(chunk_path, language_code='ja-JP'):
    client = speech.SpeechClient()
    with open(chunk_path, 'rb') as f:
        content = f.read()
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

st.markdown("<h1 style='text-align:center;'>担当者会議音声記録ツール</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>MP3またはM4Aファイルを以下にドラッグ＆ドロップまたはクリックして選択してください。</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[1]}') as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        input_file_path = tmp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    # 処理中メッセージ
    st.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # 処理時間を記録する辞書
    timing = {}

    # 入力ファイルからWAVへの変換
    start_time = time.perf_counter()
    progress_bar.progress(10)
    if convert_to_wav(input_file_path, wav_file_path):
        end_time = time.perf_counter()
        timing['ファイル変換 (WAV)'] = end_time - start_time
        progress_bar.progress(20)
        try:
            # 音声分割
            start_time = time.perf_counter()
            chunks = split_audio_to_chunks(wav_file_path, chunk_length_ms=50000)  # 50秒単位で分割
            end_time = time.perf_counter()
            timing['音声分割'] = end_time - start_time
            progress_bar.progress(30)

            # 並列処理で文字起こし
            start_time = time.perf_counter()
            transcripts = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(transcribe_chunk, chunk) for chunk in chunks]
                for i, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    transcripts.append(result)
                    # 進捗を更新 (最大30→80程度に分配)
                    current_progress = 30 + int((i / len(chunks)) * 50)
                    progress_bar.progress(min(current_progress, 80))

            full_transcript = "\n".join(transcripts)
            end_time = time.perf_counter()
            timing['音声の文字起こし(分割並列)'] = end_time - start_time

            progress_bar.progress(85)
            # OpenAI APIで話題分類
            start_time = time.perf_counter()
            # 担当者会議用のプロンプト
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは介護領域における幅広い専門知識を持つアシスタントです。特に担当者会議に関する情報に基づいて、議事録を作成することができます。情報がない場合は必ず'記載なし'と記してください。"},
                    {
                        "role": "user",
                        "content":
                        """
最下部に述べる音声記録を参考に、以下の手順で議事録の要約と内容整理を行ってください。

1. **基本情報・開催目的**
   - 利用者氏名、要介護度、主介護者、参加者氏名・所属
   - 開催年月日・場所
   - 開催の目的

2. **検討した項目**
   - 利用者・家族の希望確認、身体・生活状況の確認、利用サービス内容・回数・費用確認、各事業所の役割分担、医療的留意点、緊急時対応策など

3. **検討内容**
   - 各検討項目に対応する詳細内容

4. **結論（合意形成内容）**
   - 本日の協議内容に基づく最終方針、プラン承認の有無、サービス継続・変更の確定事項、関係者全員の合意が得られた点

5. **残された課題・次回開催について**
   - 今回解決に至らなかった点、状態観察期間、問題発生時の緊急開催要否、次回会議の目安

6. **欠席者照会内容（必要に応じて）**
   - 欠席したサービス担当者への情報提供内容・照会事項

7. **文書交付等の扱い**
   - 会議の要点を事業所に渡す義務の有無、情報共有としての提供方法

# 出力形式

- 各項目を漏れが無いように、丁寧な文章でまとめてください。
- 必ず情報がない場合は'記載なし'と明記してください。
- 形式例:
基本情報・開催目的 利用者氏名：〇〇 要介護度：記載なし 主介護者：〇〇 参加者氏名・所属：〇〇

開催年月日・場所：〇年〇月〇日、〇場所 開催の目的：〇〇

検討した項目 ① 利用者・家族の希望確認 ② 身体・生活状況の確認 ...

検討内容 ① 利用者・家族の意向： 本人『〇〇』 家族『〇〇』 ② 身体状況： 主治医意見書より〇〇 ...

結論（合意形成内容） ・〇〇 ・〇〇 ...

残された課題・次回開催について 残された課題： ・〇〇 次回会議の目安： ・〇〇

欠席者照会内容（必要に応じて） ・〇〇

文書交付等の扱い ・〇〇

css
コードをコピーする

# Notes
情報がない場合は必ず'記載なし'と記入すること。
"""
                  },
                  {"role": "user", "content": full_transcript}
              ]
          )
          topic_content = response['choices'][0]['message']['content'].strip()
          end_time = time.perf_counter()
          timing['話題分類 (OpenAI GPT-4)'] = end_time - start_time

          progress_bar.progress(100)

          # 結果の表示
          st.markdown("<h2 style='text-align:center;'>結果</h2>", unsafe_allow_html=True)

          # GPTの出力を直接表示
          st.markdown("### GPTの出力")
          st.markdown(f"<div style='padding:10px; font-size:1.2em; white-space: pre-wrap;'>{topic_content}</div>", unsafe_allow_html=True)

          # 処理時間の表示（必要に応じて保持）
          st.markdown("### 処理時間")
          timing_df = pd.DataFrame({
              "ステップ": list(timing.keys()),
              "所要時間 (秒)": [f"{v:.2f}" for v in timing.values()]
          })
          st.table(timing_df)

      except Exception as e:
          st.error(f"処理に失敗しました: {e}")

  # 一時ファイル削除
  try:
      os.remove(input_file_path)
      os.remove(wav_file_path)
      for chunk in locals().get('chunks', []):
          try:
              os.remove(chunk)
          except:
              pass
  except Exception as e:
      st.warning(f"一時ファイルの削除に失敗しました: {e}")
