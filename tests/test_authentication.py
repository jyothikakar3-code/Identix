"""Regression tests for profile changes and login aliases."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app


class AuthenticationRegressionTests(unittest.TestCase):
    def test_changed_profile_can_log_in_by_username_display_name_or_email(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            app, "DB_PATH", Path(temporary_directory) / "auth.sqlite3"
        ), patch.object(app, "DATABASE_URL", ""):
            app.init_db()
            admin = app.rows("auth_users", "WHERE username = ?", ("admin",))[0]
            new_password = "changed-password"
            app.execute(
                "UPDATE auth_users SET display_name=?, email=?, password_hash=? WHERE id=?",
                (
                    "Changed Display Name",
                    "changed@example.com",
                    app.hash_password(new_password),
                    admin["id"],
                ),
            )

            for identifier in ("admin", "Changed Display Name", "changed@example.com"):
                authenticated = app.authenticate_user(identifier, new_password)
                self.assertIsNotNone(authenticated)
                self.assertEqual(authenticated["id"], admin["id"])

            self.assertIsNone(app.authenticate_user("admin", "admin123"))
            self.assertFalse(app.default_admin_credentials_active())

    def test_duplicate_display_name_is_never_resolved_to_an_arbitrary_user(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            app, "DB_PATH", Path(temporary_directory) / "ambiguous.sqlite3"
        ), patch.object(app, "DATABASE_URL", ""):
            app.init_db()
            shared_password = "shared-password"
            app.execute(
                "UPDATE auth_users SET display_name=?, password_hash=? WHERE username='admin'",
                ("Shared Name", app.hash_password(shared_password)),
            )
            app.execute(
                "INSERT INTO auth_users(username, password_hash, role, display_name, email, phone, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    "second-user",
                    app.hash_password(shared_password),
                    "User",
                    "Shared Name",
                    "second@example.com",
                    "",
                    app.now_iso(),
                ),
            )

            self.assertIsNone(app.authenticate_user("Shared Name", shared_password))
            self.assertEqual(
                app.authenticate_user("second-user", shared_password)["username"],
                "second-user",
            )

    def test_username_normalization_and_validation(self) -> None:
        self.assertEqual(app.normalize_username("  New.Admin_1  "), "new.admin_1")
        self.assertTrue(app.valid_username("new.admin_1"))
        self.assertFalse(app.valid_username("Full Name"))
        self.assertFalse(app.valid_username("ab"))


if __name__ == "__main__":
    unittest.main()
