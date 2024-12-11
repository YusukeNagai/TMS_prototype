file_id = None  # 初期化

# ファイルアップロード処理（必要に応じて有効化）
# try:
#     with open('前学習_介護用語リスト.jsonl', 'rb') as file:
#         file_metadata = openai.File.create(
#             file=file,
#             purpose='fine-tune'
#         )
#     file_id = file_metadata['id']
#     st.write(f"事前学習用ファイルをアップロードしました。ファイルID: {file_id}")
# except Exception as e:
#     st.error(f"事前学習用ファイルのアップロードに失敗しました: {e}")

# file_idが存在する場合のみ処理を実行
if file_id:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # モデル名を修正
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
else:
    st.info("ファイルIDが存在しないため、話題分類はスキップされました。")
