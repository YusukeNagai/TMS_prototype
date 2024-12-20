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

# MP3からWAVへの変換関数
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        subprocess.run(['ffmpeg', '-y', '-i', mp3_file_path, '-ar', '16000', '-ac', '1', wav_file_path], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpegの変換に失敗しました: {e}")
        return False
    return True

# WAVファイルを分割する関数
def split_audio_to_chunks(wav_file_path, chunk_length_ms=59000):
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

    # 処理時間を記録する辞書
    timing = {}

    # MP3→WAV変換
    start_time = time.perf_counter()
    progress_bar.progress(10)
    if convert_mp3_to_wav(mp3_file_path, wav_file_path):
        end_time = time.perf_counter()
        timing['MP3 to WAV変換'] = end_time - start_time
        progress_bar.progress(20)
        try:
            # 音声分割
            start_time = time.perf_counter()
            chunks = split_audio_to_chunks(wav_file_path, chunk_length_ms=59000)  # 1分単位で分割
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
                    current_progress = 30 + int((i/len(chunks))*50)
                    progress_bar.progress(min(current_progress, 80))

            full_transcript = "\n".join(transcripts)
            end_time = time.perf_counter()
            timing['音声の文字起こし(分割並列)'] = end_time - start_time

            progress_bar.progress(85)
            # OpenAI APIで話題分類
            start_time = time.perf_counter()
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content":"あなたは介護領域における幅広い専門知識を持つアシスタントです。特にケアマネジャー向けの情報に関して専門的な回答を提供できます。"},
                    {
                        "role": "user",
                        "content":
                        """
最下部に述べる音声記録を参考に、以下の手順でテキストの要約と内容整理を行ってください。

1. **要約作成**
   - 文章全体の中から重要な内容を抽出し、簡潔に要約します。

2. **話題の項目と内容の整理**
   - 要約した内容を更に分解し、話題に合わせた項目とその具体的な内容を整理して提示してください。
   - それぞれの項目と内容は「要約: 田中さんは～」の形式で記述してください。

下記の項目は音声記録にすべて含まれているわけではありません。音声記録に上がった項目のみ分類してください。

_____________________________________________________________
-基本情報
 利用者名
 作成日
 作成者
 状態に関する項目
 1. コミュニケーション
  視力：
  聴力：
  会話能力：
  活用しているコミュニケーション機器：
  特記事項：
  維持・改善の要素や利点：
 2. 認知と行動
  認知障害：
  精神症状：
  特記事項：
  維持・改善の要素や利点：
 3. 家族・知人等の状況
  介護提供：
  介護者の負担感：
  介護者の就労・就学状況：
  特記事項：
  維持・改善の要素や利点：
 4. 健康状態
  主疾病：
  薬剤の使用状況：
  口腔内の状況：
  義歯の有無等：
  食事摂取：
  飲水量：
  栄養状態：
  身長・体重：
  血圧：
  麻痺・拘縮、皮膚・爪の問題：
  入浴：
  排泄（便・尿）：
  生活リズム：
  特記事項：
 5. ADL（日常生活動作）
  食事：
  排泄：
  入浴：
  更衣・整容：
  移動：
  特記事項：
  維持・改善の要素や利点：
 6. IADL（手段的日常生活動作）
  買い物：
  服薬状況：
  住環境：
  維持・改善の要素や利点：
 7. 社会交流
  社会参加：
  対人交流：
  特記事項：
 8. その他留意すべき事項
  虐待の可能性、経済的困窮、医療依存度、趣味や得意なことなどを記載。
  問題（困りごと）
  利用者の困りごと：
  家族の困りごと：
  意向・意見・判断
  利用者意向：
  家族意向：
  医師・専門職等意見：
  CM判断：
  解決すべき課題（ニーズ）
  整理前：
  関連：
  整理後：
  優先順位：
  気づき：

# 出力形式

- 各要約と項目の内容は短く、簡潔な文でまとめてください。
- 形式例: `要約: [該当内容]`

# 例

**入力**
こんにちは、田中さん～

**出力**
要約: 田中さんは～

コミュニケーション 視力：～

# Notes

- 記入項目ごとの注意点に従い、信頼性のある情報を選んで要約に含めてください。
- 各要約が論理的に正確であることを確認してください。
- 情報の非対称性や誤解を避けるため、明確で簡潔な表現を心がけてください。
"""
                    },
                    {"role": "user", "content": full_transcript}
                ]
            )
            topic_content = response['choices'][0]['message']['content'].strip()
            end_time = time.perf_counter()
            timing['話題分類 (OpenAI GPT-4)'] = end_time - start_time

            progress_bar.progress(100)

            # 要約部分を抽出
            lines = topic_content.split("\n")
            summary = "記載なし"
            topic_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue  # 空行をスキップ
                if line.lower().startswith("要約"):
                    if ":" in line:
                        summary = line.split(":", 1)[1].strip()
                    else:
                        summary = line
                    continue  # 要約行をスキップ
                if re.match(r'^\d+\.', line):
                    continue  # 大項目（数字とドットで始まる行）をスキップ
                topic_lines.append(line)

            categories = []
            values = []

            for line in topic_lines:
                if ":" in line:
                    c, v = line.split(":", 1)
                    c = c.strip()
                    v = v.strip()
                    if c and v:
                        categories.append(c)
                        values.append(v)
                else:
                    # コロンがない行はスキップ
                    continue

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

            # 処理時間の表示
            st.markdown("### 処理時間")
            timing_df = pd.DataFrame({
                "ステップ": list(timing.keys()),
                "所要時間 (秒)": [f"{v:.2f}" for v in timing.values()]
            })
            st.table(timing_df)

            # リマインドメッセージの表示
            st.markdown("### 今後の質問事項")
            st.markdown("次回は食事や水分補給について聞いてみると、田中さんの健康管理に関する理解が深まりそうです。")

        except Exception as e:
            st.error(f"処理に失敗しました: {e}")

    # 一時ファイル削除
    try:
        os.remove(mp3_file_path)
        os.remove(wav_file_path)
        for chunk in locals().get('chunks', []):
            try:
                os.remove(chunk)
            except:
                pass
    except Exception as e:
        st.warning(f"一時ファイルの削除に失敗しました: {e}")
