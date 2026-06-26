# Deploy Online

The app is ready for online hosting.

## Recommended: Streamlit Community Cloud

1. Go to `https://github.com` and create a new repository.
2. Upload this project folder to the repository.
3. Go to `https://share.streamlit.io`.
4. Sign in with GitHub.
5. Choose the repository.
6. Set the main file path to:

```text
app.py
```

7. Click Deploy.

## Other Hosting Options

Render, Railway, and similar Python hosts can use:

```text
Procfile
```

The app start command is:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Important

Free Streamlit hosting may reset local database and uploaded images when the app restarts.
For a real school, college, office, or enterprise deployment, use persistent storage or a
cloud database.
