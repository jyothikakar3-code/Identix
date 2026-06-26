# Start Here

Your AI Face Recognition Management System is ready.

## Open The App

Double-click:

```text
Start_App.command
```

Or run:

```bash
python3 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

## Login

```text
Username: admin
Password: admin123
```

Change the password from the Profile page after logging in.

## Main Files

- `app.py` is the complete application.
- `data/face_system.sqlite3` stores users, attendance, alerts, cameras, settings, and logs.
- `storage/registered` stores registered face samples.
- `storage/unknown` stores unknown visitor alert images.
- `AI_Face_Recognition_App.zip` is the bundled copy of the app.
