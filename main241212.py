# 必要な部分だけ抜粋
import os
import wave
import json
import streamlit as st
from google.cloud import speech
import openai
import tempfile
import subprocess
import pandas as pd

# (コード全体の前半部分省略...)

if uploaded_file is not None:
    # 音声ファイルの変換と処理の部分はそのまま...

    # 話題分類
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "以下のテキストを要約し、指定された項目に分類してください。空白の項目は「記載なし」としてください。"
                },
                {"role": "user", "content": transcribed_text}
            ]
        )
        topic_content = response['choices'][0]['message']['content'].strip()

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
        st.error(f"話題分類に失敗しました: {e}")

# (コード全体の後半部分省略...)
