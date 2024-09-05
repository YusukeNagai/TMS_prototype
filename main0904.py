import os
import wave
import streamlit as st
from google.cloud import speech
import openai

# OpenAI APIキーを環境変数から取得
openai.api_key = os.getenv('OPENAI_API_KEY')

# Google Cloud認証情報を環境変数から設定
google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if google_credentials:
    with open('google_credentials.json', 'w') as f:
        f.write(google_credentials)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_credentials.json'
else:
    st.error("Google Cloud credentials not found")

# MP3ファイルをWAVファイルに変換する関数
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    os.system(f'ffmpeg -i {mp3_file_path} {wav_file_path}')

# 音声ファイルをチャンクに分割する関数
def generate_audio_chunks(file_path, chunk_size=4096):
    with open(file_path, 'rb') as audio_file:
        while True:
            chunk = audio_file.read(chunk_size)
            if not chunk:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

# Streamlit アプリケーションの設定
st.title('音声ファイルの処理と話題分類')

uploaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")

if uploaded_file:
    mp3_file_path = 'uploaded_file.mp3'
    wav_file_path = 'uploaded_file.wav'

    with open(mp3_file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    convert_mp3_to_wav(mp3_file_path, wav_file_path)

    with wave.open(wav_file_path, 'rb') as f:
        fr = f.getframerate()

    st.write(f"サンプリングレート: {fr}")

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fr,
        language_code='ja-JP'
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config)
    requests = generate_audio_chunks(wav_file_path)
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    transcribed_text = ""
    for response in responses:
        for result in response.results:
            transcribed_text += result.alternatives[0].transcript + '\n'

    st.write("文字起こし結果:")
    st.text_area("Transcribed Text", transcribed_text, height=300)

    # 事前学習用ファイルのアップロード
    try:
        with open('前学習_介護用語リスト.txt', 'rb') as file:
            file_metadata = openai.File.create(
                file=file,
                purpose='fine-tune'
            )
        file_id = file_metadata['id']
    except Exception as e:
        st.error(f"事前学習用ファイルのアップロードに失敗しました: {e}")
        file_id = None

    if file_id:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"以下のテキストを分類してください。"},
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
