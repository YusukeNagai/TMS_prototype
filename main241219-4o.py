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
from pydub import AudioSegment, silence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

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
def convert_to_wav(input_bytes, output_file_path):
    try:
        # ffmpegを使用して入力バイトデータをWAVに変換
        with tempfile.NamedTemporaryFile(delete=False, suffix='.input') as tmp_input:
            tmp_input.write(input_bytes)
            tmp_input_path = tmp_input.name
        subprocess.run([
            'ffmpeg', '-y', '-i', tmp_input_path,
            '-ar', '16000', '-ac', '1', output_file_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp_input_path)
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpegの変換に失敗しました: {e}")
        return False
    return True

# 無音部分をトリミングする関数
def trim_silence(input_wav_path, output_wav_path, silence_thresh=-40, min_silence_len=1000):
    try:
        audio = AudioSegment.from_wav(input_wav_path)
        # 無音部分を検出して除去
        non_silence_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        if not non_silence_ranges:
            st.error("無音部分のみの音声ファイルです。")
            return False
        trimmed_audio = AudioSegment.empty()
        for start, end in non_silence_ranges:
            trimmed_audio += audio[start:end]
        trimmed_audio.export(output_wav_path, format="wav")
    except Exception as e:
        st.error(f"無音トリミングに失敗しました: {e}")
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

# ファイルアップロード部分を日本語化
uploaded_file = st.file_uploader(
    "MP3またはM4Aファイルをドラッグ＆ドロップするか、クリックして選択してください。",
    type=["mp3", "m4a"]
)

if uploaded_file is not None:
    # ファイルデータをメモリに読み込む
    input_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
        wav_file_path = tmp_wav.name

    # 処理中メッセージ
    st.markdown("<h2 style='text-align:center;'>ファイルを処理中です…</h2>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    # 処理時間を記録する辞書
    timing = {}

    try:
        # 入力ファイルからWAVへの変換
        start_time = time.perf_counter()
        progress_bar.progress(10)
        if convert_to_wav(input_bytes, wav_file_path):
            end_time = time.perf_counter()
            timing['ファイル変換 (WAV)'] = end_time - start_time
            progress_bar.progress(20)

            # 無音部分のトリミング
            start_time = time.perf_counter()
            trimmed_wav_path = wav_file_path.replace('.wav', '_trimmed.wav')
            if trim_silence(wav_file_path, trimmed_wav_path, silence_thresh=-40, min_silence_len=1000):
                timing['無音トリミング'] = time.perf_counter() - start_time
                progress_bar.progress(30)
            else:
                # トリミングに失敗した場合は元のWAVを使用
                trimmed_wav_path = wav_file_path
                timing['無音トリミング'] = 0
                progress_bar.progress(30)

            # 音声分割
            start_time = time.perf_counter()
            chunks = split_audio_to_chunks(trimmed_wav_path, chunk_length_ms=50000)  # 50秒単位で分割
            end_time = time.perf_counter()
            timing['音声分割'] = end_time - start_time
            progress_bar.progress(40)

            # Google Speech-to-Text クライアントの再利用
            client = speech.SpeechClient()

            # 並列処理で文字起こし
            start_time = time.perf_counter()
            transcripts = []
            with ThreadPoolExecutor(max_workers=10) as executor:  # スレッド数を増加
                futures = [executor.submit(transcribe_chunk, chunk, client) for chunk in chunks]
                for i, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    transcripts.append(result)
                    # 進捗を更新 (最大40→80程度に分配)
                    current_progress = 40 + int((i / len(chunks)) * 40)
                    progress_bar.progress(min(current_progress, 80))

            full_transcript = "\n".join(transcripts)
            end_time = time.perf_counter()
            timing['音声の文字起こし(並列)'] = end_time - start_time

            progress_bar.progress(85)
            # OpenAI APIで話題分類
            start_time = time.perf_counter()
            # プロンプトに「情報がない場合は必ず'記載なし'と書くこと」を明示
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは介護領域における幅広い専門知識を持つアシスタントです。特にケアマネジャー向けの情報に関して専門的な回答を提供できます。情報がない場合は必ず'記載なし'と記してください。"},
                    {
                        "role": "user",
                        "content":
                        """
最下部に述べる音声記録を参考に、以下の手順でテキストの要約と内容整理を行ってください。

1. **要約作成**
   - 文章全体の中から重要な内容を抽出し、簡潔に要約します。
   - 情報がない場合は必ず'記載なし'と書いてください。

2. **話題の項目と内容の整理**
   - 要約した内容を更に分解し、話題に合わせた項目とその具体的な内容を整理して提示してください。
   - 各項目で情報がない場合は必ず'記載なし'と書いてください。
   - それぞれの項目と内容は「要約: 田中さんは～」の形式で記述してください。

下記の項目のうち、音声記録に上がった項目を分類してください。
音声記録の内容から漏れの無いようにしなさい。
ただし音声記録の文脈や雰囲気から予測できる項目の内容があれば記載しなさい。
なるべく記載なしの項目が少なくなることが望ましいです。ただし、音声データに含まれないことや、音声データから予測できないことは、'記載なし'と明記してください。

例：
「
２認知と行動
認知障害：A（理由：○○という発言により）
特記事項：B
精神症状：記載なし
維持・改善の要素や利点：記載なし
」
ルール：記載なしの項目は末尾に書く
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

- 各要約と項目の内容は漏れが無いように、丁寧な文でまとめてください。
- 形式例:
 `要約: [該当内容]`
 ...

# 例

**入力**
こんにちは、田中さん～

**出力**
要約: 田中さんは～
コミュニケーション 視力：A
...
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

    finally:
        # 一時ファイル削除
        try:
            os.remove(wav_file_path)
        except Exception as e:
            st.warning(f"一時ファイルの削除に失敗しました: {e}")
        try:
            if 'trimmed_wav_path' in locals() and trimmed_wav_path != wav_file_path:
                os.remove(trimmed_wav_path)
        except Exception as e:
            st.warning(f"トリミング後の一時ファイルの削除に失敗しました: {e}")
