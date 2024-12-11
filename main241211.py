import os
import io
import json
import wave
import streamlit as st
from google.cloud import speech
import openai
from pydub import AudioSegment

# OpenAI APIキーをStreamlit Secretsから取得
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit SecretsからGoogle Cloud認証情報を取得
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]

# JSONデータをメモリ上で読み込む
google_credentials_json = json.dumps(google_credentials_data)

# 環境変数を設定
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_credentials.json'

# メモリ上でGoogle認証情報を保存
with open('google_credentials.json', 'w') as f:
    f.write(google_credentials_json)

# 音声ファイルをWAVフォーマットに変換する関数（メモリ内で処理）
def convert_mp3_to_wav(mp3_bytes):
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# 音声データをチャンクに分割するジェネレーター
def generate_audio_chunks(wav_bytes, chunk_size=4096):
    with io.BytesIO(wav_bytes) as audio_stream:
        while True:
            chunk = audio_stream.read(chunk_size)
            if not chunk:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

# Streamlit アプリケーションの設定
st.title('音声ファイルの処理と話題分類')

uploaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")

if uploaded_file is not None:
    try:
        # MP3ファイルをWAVに変換
        wav_io = convert_mp3_to_wav(uploaded_file.read())

        # WAVデータのサンプリングレートを取得
        with wave.open(wav_io, 'rb') as wf:
            fr = wf.getframerate()

        st.write(f"サンプリングレート: {fr} Hz")

        # Google Cloud Speechクライアントを初期化
        client = speech.SpeechClient()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=fr,
            language_code='ja-JP'
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config)

        # 音声データをチャンクに分割
        wav_io.seek(0)
        requests = generate_audio_chunks(wav_io.read())

        # ストリーミングで文字起こしを実行
        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        transcribed_text = ""
        for response in responses:
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + '\n'

        st.write("文字起こし結果:")
        st.text_area("Transcribed Text", transcribed_text, height=300)

        if transcribed_text.strip():
            try:
                # OpenAI GPT-4を使用して話題分類を実施
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "以下のテキストを下記の項目ごとに分類してください。"
                                "介護領域の特有の単語を考慮して分類してください。\n"
                                "項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, "
                                "投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, "
                                "記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, "
                                "息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, "
                                "現在の訪問介護"
                            )
                        },
                        {"role": "user", "content": transcribed_text}
                    ]
                )
                topic_content = response['choices'][0]['message']['content'].strip()
                topics = topic_content.split('\n')
                st.write('分類された話題:')
                for topic in topics:
                    st.write(f'- {topic}')
            except Exception as e:
                st.error(f"話題分類に失敗しました: {e}")
        else:
            st.info("文字起こしされたテキストが存在しないため、話題分類は実行されません。")
    except Exception as e:
        st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")
else:
    st.info("MP3ファイルをアップロードしてください。")
