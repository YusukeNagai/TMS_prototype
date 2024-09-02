import os
import wave
import streamlit as st
from google.cloud import speech
import openai

# 必要なライブラリをインストール
# pip install pydub google-cloud-speech openai
# FFmpegがインストールされていることを確認する

# APIキーの設定
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secret_key.json'
openai.api_key = 'My key'

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
    # ファイルパスを設定
    mp3_file_path = 'uploaded_file.mp3'
    wav_file_path = 'uploaded_file.wav'
    
    # アップロードされたファイルを保存
    with open(mp3_file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # MP3ファイルをWAVファイルに変換
    convert_mp3_to_wav(mp3_file_path, wav_file_path)
    
    # サンプリングレートを確認
    with wave.open(wav_file_path, 'rb') as f:
        fr = f.getframerate()
    
    st.write(f"サンプリングレート: {fr}")
    
    # 音声ファイルをストリーミング処理
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fr,
        language_code='ja-JP'
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config)
    requests = generate_audio_chunks(wav_file_path)
    responses = client.streaming_recognize(config=streaming_config, requests=requests)
    
    # 文字起こし結果を保存
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
        # GPT-4を使った話題分類
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"以下のテキストを下に指定する項目ごとに分類してください。介護領域の特有の単語リストを使用して分類してください。特有の単語リストのファイルIDは{file_id}です。項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, 投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, 記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, 息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, 現在の訪問介護"},
                    {"role": "user", "content": transcribed_text}
                ]
            )
            # 話題分類の結果を改行で分割して表示
            topic_content = response['choices'][0]['message']['content'].strip()
            topics = topic_content.split('\n')
            st.write('分類された話題:')
            for topic in topics:
                st.write(f'- {topic}')
        except Exception as e:
            st.error(f"話題分類に失敗しました: {e}")
