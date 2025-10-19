mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
maxUploadSize = 1000
" > ~/.streamlit/config.toml