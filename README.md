# AI Face Recognition Management System

Production-style Streamlit application for face detection, registration, recognition,
verification, liveness checks, attendance automation, unknown visitor alerts, camera
management, reports, analytics, role-based access, profile management, and settings.

## Run

```bash
python3 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Default administrator:

- Username: `admin`
- Password: `admin123`

Change the password from the Profile page after login.

## Storage

- Database: `data/face_system.sqlite3`
- Registered face crops: `storage/registered`
- Unknown visitor images: `storage/unknown`
- Reports: generated from the Attendance and Reports pages

## Notes

The app uses local OpenCV-based face detection and image embeddings so it can run
without cloud services. For high-security deployment, connect a dedicated face
recognition model, HTTPS authentication, encrypted backups, and organization privacy
controls.

## Deploy Online

Recommended beginner path:

1. Create a GitHub repository.
2. Upload this project folder.
3. Open Streamlit Community Cloud.
4. Choose the repository and set the main file to `app.py`.
5. Deploy.

Alternative hosts such as Render or Railway can use the included `Procfile`.
