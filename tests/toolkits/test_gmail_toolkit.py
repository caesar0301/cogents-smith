"""
Tests for GmailToolkit functionality.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def gmail_config():
    """Create a test configuration for GmailToolkit."""
    return ToolkitConfig(
        name="gmail",
        config={
            "GMAIL_ACCESS_TOKEN": "test_access_token",
            "GMAIL_CREDENTIALS_FILE": "/path/to/credentials.json",
            "GMAIL_TOKEN_FILE": "/path/to/token.json",
        },
    )


@pytest.fixture
def gmail_config_no_token():
    """Create a test configuration for GmailToolkit without access token."""
    return ToolkitConfig(
        name="gmail",
        config={
            "GMAIL_CREDENTIALS_FILE": "/path/to/credentials.json",
            "GMAIL_TOKEN_FILE": "/path/to/token.json",
        },
    )


@pytest.fixture
def gmail_toolkit(gmail_config):
    """Create GmailToolkit instance for testing."""
    try:
        return get_toolkit("gmail", gmail_config)
    except KeyError as e:
        if "gmail" in str(e):
            pytest.skip("GmailToolkit not available for testing")
        raise


@pytest.fixture
def gmail_toolkit_no_token(gmail_config_no_token):
    """Create GmailToolkit instance without access token for testing."""
    try:
        return get_toolkit("gmail", gmail_config_no_token)
    except KeyError as e:
        if "gmail" in str(e):
            pytest.skip("GmailToolkit not available for testing")
        raise


@pytest.fixture
def mock_gmail_service():
    """Create a mock Gmail API service."""
    mock_service = Mock()
    mock_message = {
        "id": "msg123",
        "threadId": "thread123",
        "internalDate": "1699123456000",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Test Email Subject"},
                {"name": "From", "value": "test@example.com"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Date", "value": "Mon, 01 Nov 2023 10:00:00 +0000"},
            ],
            "body": {"data": "VGVzdCBlbWFpbCBjb250ZW50"},  # Base64 encoded "Test email content"
        },
    }

    mock_service.users().messages().list().execute.return_value = {"messages": [{"id": "msg123"}]}
    mock_service.users().messages().get().execute.return_value = mock_message

    return mock_service


class TestGmailService:
    """Test cases for GmailService."""

    def test_gmail_service_initialization_with_access_token(self):
        """Test GmailService initialization with access token."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService(access_token="test_token")

        assert service.access_token == "test_token"
        assert service.config_dir == Path.home() / ".cogents"
        assert not service.is_authenticated()

    @patch("cogents_smith.toolkits.gmail_toolkit.Path.mkdir")
    def test_gmail_service_initialization_with_custom_paths(self, mock_mkdir):
        """Test GmailService initialization with custom paths."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService(
            credentials_file="/custom/creds.json", token_file="/custom/token.json", config_dir="/custom/config"
        )

        assert str(service.credentials_file) == "/custom/creds.json"
        assert str(service.token_file) == "/custom/token.json"
        assert service.config_dir == Path("/custom/config").expanduser().resolve()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("cogents_smith.toolkits.gmail_toolkit.build")
    @patch("cogents_smith.toolkits.gmail_toolkit.Credentials")
    async def test_authenticate_with_access_token_success(self, mock_credentials, mock_build):
        """Test successful authentication with access token."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        # Mock credentials and Gmail service
        mock_creds = Mock()
        mock_credentials.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        service = GmailService(access_token="test_token")
        result = await service.authenticate()

        assert result is True
        assert service.is_authenticated()
        assert service.service == mock_service
        mock_credentials.assert_called_once_with(token="test_token", scopes=service.SCOPES)

    @patch("cogents_smith.toolkits.gmail_toolkit.build")
    @patch("cogents_smith.toolkits.gmail_toolkit.Credentials")
    async def test_authenticate_with_access_token_failure(self, mock_credentials, mock_build):
        """Test authentication failure with access token."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        # Mock credentials to raise exception
        mock_build.side_effect = Exception("Authentication failed")

        service = GmailService(access_token="bad_token")
        result = await service.authenticate()

        assert result is False
        assert not service.is_authenticated()

    @patch("os.path.exists")
    @patch("cogents_smith.toolkits.gmail_toolkit.Credentials.from_authorized_user_file")
    @patch("cogents_smith.toolkits.gmail_toolkit.build")
    async def test_authenticate_with_existing_tokens(self, mock_build, mock_from_file, mock_exists):
        """Test authentication with existing valid tokens."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        # Mock existing token file and valid credentials
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = True
        mock_from_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service

        service = GmailService()
        result = await service.authenticate()

        assert result is True
        assert service.is_authenticated()

    async def test_get_recent_emails_not_authenticated(self):
        """Test getting emails when not authenticated."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService()
        emails = await service.get_recent_emails()

        assert emails == []

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    async def test_get_recent_emails_success(self, mock_is_auth, mock_gmail_service):
        """Test successful email retrieval."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        mock_is_auth.return_value = True

        service = GmailService()
        service.service = mock_gmail_service

        emails = await service.get_recent_emails(max_results=1, query="test", time_filter="1h")

        assert len(emails) == 1
        assert emails[0]["id"] == "msg123"
        assert emails[0]["subject"] == "Test Email Subject"
        assert emails[0]["from"] == "test@example.com"
        assert "Test email content" in emails[0]["body"]

    def test_parse_email(self):
        """Test email parsing functionality."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService()

        message = {
            "id": "msg123",
            "threadId": "thread123",
            "internalDate": "1699123456000",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "sender@test.com"},
                    {"name": "Date", "value": "Mon, 01 Nov 2023 10:00:00 +0000"},
                ],
                "body": {"data": "VGVzdCBjb250ZW50"},  # "Test content" in base64
            },
        }

        parsed = service._parse_email(message)

        assert parsed["id"] == "msg123"
        assert parsed["subject"] == "Test Subject"
        assert parsed["from"] == "sender@test.com"
        assert "Test content" in parsed["body"]

    def test_extract_body_simple(self):
        """Test extracting body from simple email."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService()

        payload = {"body": {"data": "VGVzdCBjb250ZW50"}}  # "Test content" in base64

        body = service._extract_body(payload)
        assert "Test content" in body

    def test_extract_body_multipart(self):
        """Test extracting body from multipart email."""
        from cogents_smith.toolkits.gmail_toolkit import GmailService

        service = GmailService()

        payload = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "VGVzdCBwbGFpbiB0ZXh0"}},  # "Test plain text" in base64
                {
                    "mimeType": "text/html",
                    "body": {"data": "PGgxPlRlc3QgSFRNTDwvaDE+"},  # "<h1>Test HTML</h1>" in base64
                },
            ]
        }

        body = service._extract_body(payload)
        assert "Test plain text" in body


class TestGmailToolkit:
    """Test cases for GmailToolkit."""

    async def test_toolkit_initialization(self, gmail_toolkit):
        """Test that GmailToolkit initializes correctly."""
        assert gmail_toolkit is not None
        assert hasattr(gmail_toolkit, "authenticate_gmail")
        assert hasattr(gmail_toolkit, "get_recent_emails")
        assert hasattr(gmail_toolkit, "search_emails")
        assert hasattr(gmail_toolkit, "get_verification_codes")

    async def test_toolkit_initialization_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {"GMAIL_ACCESS_TOKEN": "env_token", "GMAIL_CREDENTIALS_PATH": "/env/creds.json"}):
            toolkit = get_toolkit("gmail", ToolkitConfig(name="gmail"))
            assert toolkit.gmail_service.access_token == "env_token"

    async def test_get_tools_map(self, gmail_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await gmail_toolkit.get_tools_map()

        expected_tools = ["authenticate_gmail", "get_recent_emails", "search_emails", "get_verification_codes"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.authenticate")
    async def test_authenticate_gmail_success(self, mock_auth, gmail_toolkit):
        """Test successful Gmail authentication."""
        mock_auth.return_value = True

        result = await gmail_toolkit.authenticate_gmail()

        assert "✅ Successfully authenticated" in result
        mock_auth.assert_called_once()

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.authenticate")
    async def test_authenticate_gmail_failure(self, mock_auth, gmail_toolkit):
        """Test Gmail authentication failure."""
        mock_auth.return_value = False

        result = await gmail_toolkit.authenticate_gmail()

        assert "❌ Failed to authenticate" in result

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.authenticate")
    async def test_authenticate_gmail_exception(self, mock_auth, gmail_toolkit):
        """Test Gmail authentication with exception."""
        mock_auth.side_effect = Exception("Connection error")

        result = await gmail_toolkit.authenticate_gmail()

        assert "❌ Gmail authentication error" in result
        assert "Connection error" in result

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_recent_emails_success(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test successful email retrieval."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = [
            {
                "id": "msg1",
                "from": "test@example.com",
                "subject": "Test Email",
                "date": "Mon, 01 Nov 2023 10:00:00 +0000",
                "body": "This is a test email content.",
            }
        ]

        result = await gmail_toolkit.get_recent_emails(keyword="test", max_results=5, time_filter="1h")

        assert "📧 Found 1 recent email" in result
        assert "Test Email" in result
        assert "test@example.com" in result
        mock_get_emails.assert_called_once_with(max_results=5, query="newer_than:1h test", time_filter="1h")

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.authenticate")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_recent_emails_auto_authenticate(self, mock_get_emails, mock_auth, mock_is_auth, gmail_toolkit):
        """Test automatic authentication when not authenticated."""
        mock_is_auth.return_value = False
        mock_auth.return_value = True
        mock_get_emails.return_value = []

        await gmail_toolkit.get_recent_emails()

        mock_auth.assert_called_once()
        mock_get_emails.assert_called_once()

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.authenticate")
    async def test_get_recent_emails_auth_failure(self, mock_auth, mock_is_auth, gmail_toolkit):
        """Test email retrieval when authentication fails."""
        mock_is_auth.return_value = False
        mock_auth.return_value = False

        result = await gmail_toolkit.get_recent_emails()

        assert "❌ Failed to authenticate with Gmail" in result

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_recent_emails_no_results(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test email retrieval with no results."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = []

        result = await gmail_toolkit.get_recent_emails(keyword="nonexistent")

        assert "📭 No recent emails found" in result
        assert "matching 'nonexistent'" in result

    @pytest.mark.parametrize(
        "max_results,expected",
        [
            (0, 1),  # Should clamp to minimum 1
            (25, 25),  # Normal value
            (100, 50),  # Should clamp to maximum 50
        ],
    )
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_recent_emails_parameter_validation(
        self, mock_get_emails, mock_is_auth, gmail_toolkit, max_results, expected
    ):
        """Test parameter validation for max_results."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = []

        await gmail_toolkit.get_recent_emails(max_results=max_results)

        # Check that the clamped value was used
        call_args = mock_get_emails.call_args[1]
        assert call_args["max_results"] == expected

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_search_emails_success(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test successful email search."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = [
            {
                "from": "security@example.com",
                "subject": "Security Alert",
                "date": "Mon, 01 Nov 2023 10:00:00 +0000",
                "body": "Your account has been accessed from a new device.",
            }
        ]

        result = await gmail_toolkit.search_emails(query="from:security@example.com", max_results=10)

        assert "🔍 Found 1 email" in result
        assert "Security Alert" in result
        mock_get_emails.assert_called_once_with(max_results=10, query="from:security@example.com", time_filter="30d")

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_search_emails_with_time_filter(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test email search with time filter."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = []

        await gmail_toolkit.search_emails(query="subject:verification", time_filter="1h")

        call_args = mock_get_emails.call_args[1]
        assert call_args["query"] == "newer_than:1h subject:verification"
        assert call_args["time_filter"] == "1h"

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_verification_codes_success(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test successful verification code extraction."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = [
            {
                "from": "noreply@github.com",
                "subject": "GitHub verification code",
                "date": "Mon, 01 Nov 2023 10:00:00 +0000",
                "body": "Your verification code is 123456. Please enter this code to continue.",
            },
            {
                "from": "security@google.com",
                "subject": "Google 2FA Code",
                "date": "Mon, 01 Nov 2023 09:50:00 +0000",
                "body": "Your OTP: 789012",
            },
        ]

        result = await gmail_toolkit.get_verification_codes(sender_keyword="github")

        assert "🔐 Found" in result
        assert "123456" in result
        # Both emails should be processed since they contain verification keywords
        assert "GitHub verification code" in result

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_verification_codes_no_codes_found(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test verification code extraction with no codes found."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = [
            {
                "from": "security@example.com",
                "subject": "Security Alert",
                "date": "Mon, 01 Nov 2023 10:00:00 +0000",
                "body": "We detected unusual activity on your account. Please review your recent login attempts.",
            }
        ]

        result = await gmail_toolkit.get_verification_codes()

        # This email contains security keywords but no extractable codes
        assert "Found 1 verification emails but no codes could be extracted" in result

    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.is_authenticated")
    @patch("cogents_smith.toolkits.gmail_toolkit.GmailService.get_recent_emails")
    async def test_get_verification_codes_filters_common_numbers(self, mock_get_emails, mock_is_auth, gmail_toolkit):
        """Test that common numbers are filtered out from verification codes."""
        mock_is_auth.return_value = True
        mock_get_emails.return_value = [
            {
                "from": "test@example.com",
                "subject": "Your code for 2024",
                "date": "Mon, 01 Nov 2023 10:00:00 +0000",
                "body": "The year 2024 is approaching. Your verification code: 567890",
            }
        ]

        result = await gmail_toolkit.get_verification_codes()

        # Should include the actual code but not the year
        assert "567890" in result
        assert "2024" not in result.split("Code 1:")[1].split("\n")[0] if "Code 1:" in result else True

    async def test_toolkit_error_handling(self, gmail_toolkit):
        """Test error handling in toolkit methods."""
        with patch.object(gmail_toolkit.gmail_service, "authenticate", side_effect=Exception("Network error")):
            result = await gmail_toolkit.get_recent_emails()
            assert "❌ Error getting recent emails" in result
            assert "Network error" in result


# Integration-style tests (would require actual Gmail API access)
class TestGmailToolkitIntegration:
    """Integration tests for GmailToolkit (require Gmail API access)."""

    @pytest.mark.integration
    async def test_real_gmail_authentication(self):
        """Test with real Gmail credentials (requires setup)."""
        # This would require actual Gmail API credentials
        config = ToolkitConfig(name="gmail", config={"GMAIL_ACCESS_TOKEN": "your_real_access_token_here"})
        toolkit = get_toolkit("gmail", config)

        result = await toolkit.authenticate_gmail()
        assert "✅ Successfully authenticated" in result

    @pytest.mark.integration
    async def test_real_email_retrieval(self):
        """Test with real email retrieval (requires Gmail access)."""
        config = ToolkitConfig(name="gmail", config={"GMAIL_ACCESS_TOKEN": "your_real_access_token_here"})
        toolkit = get_toolkit("gmail", config)

        # Authenticate first
        auth_result = await toolkit.authenticate_gmail()
        assert "✅ Successfully authenticated" in auth_result

        # Get recent emails
        result = await toolkit.get_recent_emails(max_results=1, time_filter="1d")
        assert isinstance(result, str)
        # Should either find emails or indicate no emails found
        assert ("Found" in result) or ("No recent emails" in result)

    @pytest.mark.integration
    async def test_real_verification_code_extraction(self):
        """Test real verification code extraction (requires Gmail access)."""
        config = ToolkitConfig(name="gmail", config={"GMAIL_ACCESS_TOKEN": "your_real_access_token_here"})
        toolkit = get_toolkit("gmail", config)

        result = await toolkit.get_verification_codes(time_filter="1d")
        assert isinstance(result, str)
        # Should handle the case gracefully whether codes are found or not
        assert ("Found" in result) or ("No verification emails" in result)
