
"""Test personal and collaborative project authorization rules."""

from src.storage.test_authorization import TestAuthorizationService


def test_personal_owner_can_upload() -> None:
    """Allow a user to upload into their own personal space."""

    authorization = TestAuthorizationService(
        personal_users={"user_a"}
    )

    authorization.authorize_scope_access(
        "user_a",
        "personal",
        "user_a",
    )


def test_personal_user_cannot_access_another_space() -> None:
    """Reject access to another user's personal knowledge space."""

    authorization = TestAuthorizationService(
        personal_users={"user_a", "user_b"}
    )

    try:
        authorization.authorize_scope_access(
            "user_a",
            "personal",
            "user_b",
        )
    except PermissionError:
        return

    raise AssertionError("Another user's personal space was accessible.")


def test_project_member_can_upload() -> None:
    """Allow an authorised project member to upload documents."""

    authorization = TestAuthorizationService(
        project_members={"project_1": {"user_a", "user_b"}},
        project_owners={"project_1": "user_a"},
    )

    authorization.authorize_scope_access(
        "user_b",
        "project",
        "project_1",
    )


def test_non_member_cannot_upload() -> None:
    """Reject uploads from users outside the project."""

    authorization = TestAuthorizationService(
        project_members={"project_1": {"user_a"}},
        project_owners={"project_1": "user_a"},
    )

    try:
        authorization.authorize_scope_access(
            "user_b",
            "project",
            "project_1",
        )
    except PermissionError:
        return

    raise AssertionError("A non-member was allowed to upload.")


def test_project_owner_can_delete_any_project_document() -> None:
    """Allow the project owner to delete a member's document."""

    authorization = TestAuthorizationService(
        project_members={"project_1": {"user_a", "user_b"}},
        project_owners={"project_1": "user_a"},
    )

    document = {
        "scope_type": "project",
        "scope_id": "project_1",
        "uploaded_by": "user_b",
    }

    authorization.authorize_document_deletion(
        "user_a",
        document,
    )


def test_project_member_cannot_delete_project_document() -> None:
    """Reject deletion by a project member who is not the owner."""

    authorization = TestAuthorizationService(
        project_members={"project_1": {"user_a", "user_b"}},
        project_owners={"project_1": "user_a"},
    )

    document = {
        "scope_type": "project",
        "scope_id": "project_1",
        "uploaded_by": "user_b",
    }

    try:
        authorization.authorize_document_deletion(
            "user_b",
            document,
        )
    except PermissionError:
        return

    raise AssertionError("A non-owner was allowed to delete the document.")


def test_personal_owner_can_delete_own_document() -> None:
    """Allow the owner to delete a document from their personal space."""

    authorization = TestAuthorizationService(
        personal_users={"user_a"}
    )

    document = {
        "scope_type": "personal",
        "scope_id": "user_a",
        "uploaded_by": "user_a",
    }

    authorization.authorize_document_deletion(
        "user_a",
        document,
    )


def test_personal_user_cannot_delete_another_users_document() -> None:
    """Reject deletion from another user's personal space."""

    authorization = TestAuthorizationService(
        personal_users={"user_a", "user_b"}
    )

    document = {
        "scope_type": "personal",
        "scope_id": "user_b",
        "uploaded_by": "user_b",
    }

    try:
        authorization.authorize_document_deletion(
            "user_a",
            document,
        )
    except PermissionError:
        return

    raise AssertionError("A user deleted another user's document.")


def main() -> None:
    """Run all authorization tests."""

    tests = [
        test_personal_owner_can_upload,
        test_personal_user_cannot_access_another_space,
        test_project_member_can_upload,
        test_non_member_cannot_upload,
        test_project_owner_can_delete_any_project_document,
        test_project_member_cannot_delete_project_document,
        test_personal_owner_can_delete_own_document,
        test_personal_user_cannot_delete_another_users_document,
    ]

    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")

    print(f"\nAll {len(tests)} authorization tests passed.")


if __name__ == "__main__":
    main()
