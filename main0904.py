name: Streamlit App Deployment

on:
  push:
    branches:
      - main  # mainブランチにコードをプッシュしたときに自動実行される

jobs:
  deploy:
    runs-on: ubuntu-latest  # GitHub ActionsはUbuntuの最新環境で動作する

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2  # リポジトリのコードをチェックアウト
      
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'  # Python 3.9を使用

    - name: Install ffmpeg
      run: sudo apt-get install -y ffmpeg  # ffmpegのインストール

    # シークレットを使用してJSONファイルを作成
    - name: create-json
      id: create-json
      uses: jsdaniell/create-json@v1.2.2
      with:
        name: "google_credentials.json"  # 作成するファイル名
        json: ${{ secrets.GOOGLE_APPLICATION_CREDENTIALS }}  # シークレットからJSONの内容を取得

    - name: Install distutils
      run: sudo apt-get install python3-distutils

    - name: Install dependencies
      run: |
        pip install -r requirements.txt  # プロジェクトの依存パッケージをインストール

    - name: Cache pip dependencies
      uses: actions/cache@v2
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Run Streamlit app
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # GitHub SecretsからAPIキーを取得して環境変数として設定
        GOOGLE_APPLICATION_CREDENTIALS: google_credentials.json  # 作成したGoogle Cloud認証情報ファイルのパスを設定
      run: |
        streamlit run main0904.py  # Streamlitアプリを起動
