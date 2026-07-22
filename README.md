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

- Production database: PostgreSQL through the `DATABASE_URL` Streamlit secret
- Local-development fallback: `data/face_system.sqlite3`
- Registered face crops: `storage/registered`
- Unknown visitor images: `storage/unknown`
- Reports: generated from the Attendance and Reports pages

## Notes

The app uses local OpenCV-based face detection and image embeddings so it can run
without cloud services. For high-security deployment, connect a dedicated face
recognition model, HTTPS authentication, encrypted backups, and organization privacy
controls.

## Human/animal safety gate

All face workflows call the same gate in `utils/human_face_validation.py`:

1. YuNet (or the eye-gated Haar fallback) finds face-shaped candidates.
2. The local EfficientNet-Lite4 ImageNet classifier checks each candidate and the full
   image when no human candidate exists.
3. Only detections carrying the `human_animal_guard_v1` marker may reach SFace.

Registration, recognition, verification, liveness, automatic attendance, and
unknown-visitor alerts therefore cannot create, compare, store, or act on an
animal embedding. Mixed images keep validated humans and ignore animal regions.
The model file is checksum-verified before OpenCV loads it, and model failure is
fail-closed.

The default thresholds can be calibrated with a labelled deployment dataset via
`ANIMAL_GUARD_TOP1_MIN_PROBABILITY`, `ANIMAL_GUARD_TOP5_MIN_MASS`,
`ANIMAL_GUARD_MIN_MARGIN`, `ANIMAL_GUARD_NON_FACE_OBJECT_MIN_PROBABILITY`, and
`ANIMAL_GUARD_HIGH_CONFIDENCE_FACE_OVERRIDE`. Do not claim a 95% production rate
until the exact camera/domain test set reaches it; the automated suite verifies
control flow and bypass prevention, while statistical accuracy requires labelled
human, animal, and invalid images representative of the deployment.

To extend or replace the classifier, update the model URL, checksum, preprocessing,
and `ANIMAL_CLASS_IDS` in that module. Do not add workflow-specific animal rules.
Use `scripts/evaluate_animal_guard.py` with the labelled folder structure described
in `tests/fixtures/animal_guard/README.md`; it fails on any animal false accept or
when overall accuracy is below 95%.

For Streamlit Community Cloud, configure a persistent PostgreSQL connection in
**Manage app → Settings → Secrets**:

```toml
DATABASE_URL = "postgresql://..."
```

The application creates its tables automatically. Never commit this credential
to GitHub. Without `DATABASE_URL`, local SQLite data can be replaced when a cloud
deployment restarts.

## Deploy Online

Recommended beginner path:

1. Create a GitHub repository.
2. Upload this project folder.
3. Open Streamlit Community Cloud.
4. Choose the repository and set the main file to `app.py`.
5. Deploy.

Alternative hosts such as Render or Railway can use the included `Procfile`.
