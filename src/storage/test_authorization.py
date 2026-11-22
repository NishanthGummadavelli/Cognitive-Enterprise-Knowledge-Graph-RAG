
"""Provide controlled authorization for document service tests."""

from collections.abc import Mapping
from typing import Any


class TestAuthorizationService:
    """Simulate personal ownership and project membership for tests."""

    def __init__(
        self,
        personal_users: set[str] | None = None,
        project_members: dict[str, set[str]] | None = None,
        project_owners: dict[str, str] | None = None,
    ) -> None:
        """Initialize the test users, project members and project owners."""

        self.personal_users = personal_users or set()
        self.project_members = project_members or {}
        self.project_owners = project_owners or {}

    def authorize_scope_access(
        self,
        user_id: str,
        scope_type: str,
        scope_id: str,
    ) -> None:
        """Verify personal ownership or project membership."""

        if scope_type == "personal":
            if user_id != scope_id or user_id not in self.personal_users:
                raise PermissionError(
                    "User does not own this personal knowledge space."
                )
            return

        if scope_type == "project":
            members = self.project_members.get(scope_id, set())

            if user_id not in members:
                raise PermissionError(
                    "User is not a member of this project."
                )
            return

        raise PermissionError("Unsupported document scope.")

    def authorize_document_deletion(
        self,
        user_id: str,
        document: Mapping[str, Any],
    ) -> None:
        """Allow personal owners and project owners to delete documents."""

        scope_type = document["scope_type"]
        scope_id = document["scope_id"]

        if scope_type == "personal":
            if user_id != scope_id:
                raise PermissionError(
                    "Only the personal-space owner can delete this document."
                )
            return

        if scope_type == "project":
            project_owner = self.project_owners.get(scope_id)

            if user_id != project_owner:
                raise PermissionError(
                    "Only the project owner can delete project documents."
                )
            return

        raise PermissionError("Unsupported document scope.")
