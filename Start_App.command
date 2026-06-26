#!/bin/zsh
cd "$(dirname "$0")"
echo "Starting AI Face Recognition Management System..."
echo "Open http://127.0.0.1:8501 in your browser."
python3 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
