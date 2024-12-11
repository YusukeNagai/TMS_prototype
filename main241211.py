import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai

# OpenAI APIキーをStreamlit Secretsから取得
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit SecretsからGoogle Cloud認証情報を取得
google_credentials_data = st.secrets["GOOGLE_CREDENTIALS"]

# JSONファイルとして書き出し
with open('google_credentials.json', 'w') as f:
    json.dump(dict(google_credentials_data), f)

# 環境変数を設定
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_credentials.json'

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

    # 以下、事前学習用ファイルのアップロードおよび関連処理をコメントアウト
    """
    # 事前学習用ファイルのアップロード
    file_id = None  # 初期化
    try:
        with open('前学習_介護用語リスト.jsonl', 'rb') as file:  # JSONLファイルを指定
            file_metadata = openai.File.create(
                file=file,
                purpose='fine-tune'
            )
        file_id = file_metadata['id']
        st.write(f"事前学習用ファイルをアップロードしました。ファイルID: {file_id}")
    except Exception as e:
        st.error(f"事前学習用ファイルのアップロードに失敗しました: {e}")
    """

    # file_idが存在する場合のみ処理を実行
    # 以下の処理もコメントアウトまたは削除
    """
    if file_id:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # モデル名を修正
                messages=[
                    {"role": "system", "content": f"以下のテキストを下に指定する項目ごとに分類してください。介護領域の特有の単語リストを使用して分類してください。特有の単語リストのファイルIDは{file_id}です。項目：名前, 年齢, 性別, 住所, 既往歴, 現在の状態, 医師の診断, 投薬, 住環境, 同居家族, 経済状況, 自立度, 食事, トイレ, 認知機能の状態, 記憶, 認知テスト, 趣味, 外出頻度, 友人関係, 妻の支援状況, 息子の支援状況, 一人での外出の傾向, 注意点, 要望, 現在のデイサービス, 現在の訪問介護"},
                    {"role": "user", "content": transcribed_text}
            )
            topic_content = response['choices'][0]['message']['content'].strip()
            topics = topic_content.split('\n')
            st.write('分類された話題:')
            for topic in topics:
                st.write(f'- {topic}')
        except Exception as e:
            st.error(f"話題分類に失敗しました: {e}")
    else:
        st.info("ファイルIDが存在しないため、話題分類はスキップされました。")
    """

    # **代替案として、事前学習を行わずに直接話題分類を実行する**
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 正しいモデル名を使用
            messages=[
                {"role": "system", "content": "以下のテキストを分類してください。"},
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

