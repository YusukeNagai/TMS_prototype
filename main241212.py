import streamlit as st

# CSSでfile_uploaderのスタイルを調整
st.markdown("""
<style>
/* file_uploaderの全体を大きくカスタマイズ */
div[data-testid="stFileUploader"] {
    text-align: center;
    border: 4px dashed #ccc; /* 点線の枠 */
    border-radius: 15px;    /* 角を丸く */
    padding: 50px;          /* 内側の余白を広げる */
    margin: 20px auto;      /* 周囲の余白を中央揃え */
    font-size: 1.5em;       /* 文字サイズを大きく */
    color: #333;            /* テキストの色 */
    background-color: #f9f9f9; /* 背景色 */
    width: 80%;             /* 全体幅 */
    max-width: 600px;       /* 最大幅を設定 */
    cursor: pointer;        /* ホバーで手のアイコンに */
}

/* file_uploaderがhoverされた時の効果 */
div[data-testid="stFileUploader"]:hover {
    background-color: #eee; /* 背景色を少し変更 */
    border-color: #aaa;     /* 枠線の色を濃く */
}
</style>
""", unsafe_allow_html=True)

# ファイルアップロード
st.markdown("<h1 style='text-align:center;'>ファイルアップロード</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("ここにファイルをドラッグ＆ドロップまたはクリックして選択してください", type=["mp3", "wav"])

# アップロードされたファイルの処理
if uploaded_file is not None:
    st.markdown("### アップロードされたファイル")
    st.write(f"ファイル名: {uploaded_file.name}")
    st.audio(uploaded_file)  # 音声ファイルの再生
else:
    st.markdown("<p style='text-align:center; color: gray;'>ファイルをアップロードしてください。</p>", unsafe_allow_html=True)
