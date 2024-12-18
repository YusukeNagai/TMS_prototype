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
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """最下部に記す音声記録を参考にし、以下の手順でテキストの要約と内容整理を行ってください。
                                    1. **要約作成**
                                       - 文章全体の中から重要な内容を抽出し、簡潔に要約します。

                                    2. **話題の項目と内容の整理**
                                        - 要約した内容を更に分解し、話題に合わせた項目とその具体的な内容を整理して提示してください。
                                        - それぞれの項目と内容は「要約: 田中さんは～」の形式で記述してください。

                                    下記の項目は音声記録にすべて入っているわけではない。下記の項目の中から話題に上がったもののみ、分類せよ。

                                    _____________________________________________________________
                                    -基本情報
                                     利用者名
                                     作成日
                                     作成した日付を記入する。
                                     作成者
                                     シートを作成した介護支援専門員の氏名をフルネームで記載する。
                                     状態に関する項目
                                    1. コミュニケーション
                                    視力：視力の状態（正常、弱視、失明など）
                                    聴力：聴力の状態（正常、難聴など）
                                    会話能力：話すことができる、言葉を発するのが困難、無言症など
                                    活用しているコミュニケーション機器：補聴器、コミュニケーションボードなどの利用の有無。
                                    特記事項：上記に関連する内容で特筆すべき事項を記載。
                                    維持・改善の要素や利点：利用者が取り組んでいるリハビリや日常生活での努力、またはポジティブな特徴を具体的に記載。
                                    2. 認知と行動
                                    認知障害：該当する状態を記録（軽度認知障害、認知症など）。
                                    精神症状：気分障害、不安症状、幻覚・妄想などの有無を記載。
                                    特記事項：認知面や精神面において観察された特筆事項を記載。
                                    維持・改善の要素や利点：認知症リハビリの効果、穏やかな態度などを具体的に記載。
                                    3. 家族・知人等の状況
                                    介護提供：誰が介護を提供しているのか、どのようなサポートを行っているかを記載。
                                    介護者の負担感：介護者が感じている負担の程度を記載。
                                    介護者の就労・就学状況：介護者が仕事や学校に通っている場合、その状況を特記。
                                    特記事項：家族の支援状況、介護者の健康状態などを記載。
                                    維持・改善の要素や利点：家族間の協力体制、介護者の良い面などを具体的に記載。
                                    4. 健康状態
                                    主疾病：利用者が抱える主な疾病名と症状を記載。
                                    薬剤の使用状況：どの病気に対してどの薬が処方されているかを詳細に記載。
                                    口腔内の状況：歯の本数、噛み合わせ、口腔衛生の状態を記載。
                                    義歯の有無等：義歯の利用状況を記載。
                                    食事摂取：食事の形態（普通食、ミキサー食、経管栄養など）を記載。
                                    飲水量：1日の飲水量、医師の指示がある場合はそれも記載。
                                    栄養状態：医師の意見を基に記載。
                                    身長・体重：6か月間での体重変動があれば記載。
                                    血圧：測定時の値および変動状況を記載。
                                    麻痺・拘縮、皮膚・爪の問題：該当する状態を記載。
                                    入浴：週の入浴回数を記載。
                                    排泄（便・尿）：便秘や下痢、尿意の有無などを記載。
                                    生活リズム：睡眠時間帯、日常的な活動内容を記号で記載。
                                    特記事項：健康状態に関して追加情報があれば記載。
                                    5. ADL（日常生活動作）
                                    食事：介助の必要性、使用している器具を記載。
                                    排泄：失禁の有無、排泄方法（おむつ使用など）。
                                    入浴：入浴場所、方法、介助者などを記載。
                                    更衣・整容：衣服の着脱や整容の能力を記載。
                                    移動：歩行や車椅子の使用状況を記載。
                                    特記事項：利用者のADLにおける重要な情報を記載。
                                    維持・改善の要素や利点：ポジティブな取り組みや努力を記載。
                                    6. IADL（手段的日常生活動作）
                                    買い物：支援の有無、頻度などを記載。
                                    服薬状況：服薬の管理方法を記載。
                                    住環境：居住状況の詳細を記載。
                                    維持・改善の要素や利点：自立性や工夫について記載。
                                    7. 社会交流
                                    社会参加：ボランティアや地域活動への参加状況。
                                    対人交流：家族や友人との交流状況。
                                    特記事項：社会的関係における具体例を記載。
                                    8. その他留意すべき事項
                                    虐待の可能性、経済的困窮、医療依存度、趣味や得意なことなどを記載。
                                    問題（困りごと）
                                    利用者の困りごと：利用者が表明した「～で困る」を具体的に記載。
                                    家族の困りごと：同様に家族が抱える問題を記載。
                                    意向・意見・判断
                                    利用者意向：利用者が表明した「～したい」などの肯定的な内容をそのまま記載。
                                    家族意向：家族が望む「～になってほしい」を記載。
                                    医師・専門職等意見：チームメンバーの具体的な意見を記載。
                                    CM判断：ケアマネが必要と判断した内容を「～が必要」と記載。
                                    解決すべき課題（ニーズ）
                                    整理前：利用者意向と判断が一致する内容。
                                    関連：関連するニーズの番号。
                                    整理後：最も重要なニーズを記載。
                                    優先順位
                                    ニーズを緊急性や重要性に基づいて優先順位付けする。
                                    気づき
                                    シート作成を通じて気づいたことを記載する。
                                   _____________________________________________________________
                                    # 出力形式
                                    
                                    - 各要約と項目の内容は短く、簡潔な文でまとめてください。
                                    - 形式例: `要約: [該当内容]`
                                    
                                    # 例
                                    
                                    **入力**
                                    ```
                                    こんにちは、田中さん～
                                    ```
                                    
                                    **出力**
                                    ```
                                    要約: 田中さんは～（実際に要約する際は、内容や項目の詳細情報を記載します）
                                    1. コミュニケーション
                                    視力：～
                                    ```
                                    
                                    # Notes
                                    
                                    - 記入項目ごとの注意点に従い、信頼性のある情報を選んで要約に含めてください。
                                    - 各要約が論理的に正確であることを確認してください。
                                    - 情報の非対称性や誤解を避けるため、明確で簡潔な表現を心がけてください。
                                    
                                    
                                    """
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
