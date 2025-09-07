"""
Gmail toolkit for email reading and authentication.

Provides tools for Gmail API integration including 2FA code retrieval,
email reading, and authentication management. Based on the browser-use
Gmail integration but adapted for the Cogents toolkit framework.
"""

import base64
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiofiles
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = get_logger(__name__)


class GmailService:
    """
    Gmail API service for email reading.
    Provides functionality to:
    - Authenticate with Gmail API using OAuth2
    - Read recent emails with filtering
    - Return full email content for agent analysis
    """

    # Gmail API scopes
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(
        self,
        credentials_file: str | None = None,
        token_file: str | None = None,
        config_dir: str | None = None,
        access_token: str | None = None,
    ):
        """
        Initialize Gmail Service
        Args:
            credentials_file: Path to OAuth credentials JSON from Google Cloud Console
            token_file: Path to store/load access tokens
            config_dir: Directory to store config files (defaults to ~/.cogents)
            access_token: Direct access token (skips file-based auth if provided)
        """
        # Set up configuration directory
        if config_dir is None:
            self.config_dir = Path.home() / ".cogents"
        else:
            self.config_dir = Path(config_dir).expanduser().resolve()

        # Ensure config directory exists (only if not using direct token)
        if access_token is None:
            self.config_dir.mkdir(parents=True, exist_ok=True)

        # Set up credential paths
        self.credentials_file = credentials_file or self.config_dir / "gmail_credentials.json"
        self.token_file = token_file or self.config_dir / "gmail_token.json"

        # Direct access token support
        self.access_token = access_token

        self.service = None
        self.creds = None
        self._authenticated = False

    def is_authenticated(self) -> bool:
        """Check if Gmail service is authenticated"""
        return self._authenticated and self.service is not None

    async def authenticate(self) -> bool:
        """
        Handle OAuth authentication and token management
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            logger.info("🔐 Authenticating with Gmail API...")

            # Check if using direct access token
            if self.access_token:
                logger.info("🔑 Using provided access token")
                # Create credentials from access token
                self.creds = Credentials(token=self.access_token, scopes=self.SCOPES)
                # Test token validity by building service
                self.service = build("gmail", "v1", credentials=self.creds)
                self._authenticated = True
                logger.info("✅ Gmail API ready with access token!")
                return True

            # Original file-based authentication flow
            # Try to load existing tokens
            if os.path.exists(self.token_file):
                self.creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
                logger.debug("📁 Loaded existing tokens")

            # If no valid credentials, run OAuth flow
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    logger.info("🔄 Refreshing expired tokens...")
                    self.creds.refresh(Request())
                else:
                    logger.info("🌐 Starting OAuth flow...")
                    if not os.path.exists(self.credentials_file):
                        logger.error(
                            f"❌ Gmail credentials file not found: {self.credentials_file}\n"
                            "Please download it from Google Cloud Console:\n"
                            "1. Go to https://console.cloud.google.com/\n"
                            "2. APIs & Services > Credentials\n"
                            "3. Download OAuth 2.0 Client JSON\n"
                            f"4. Save as 'gmail_credentials.json' in {self.config_dir}/"
                        )
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), self.SCOPES)
                    # Use specific redirect URI to match OAuth credentials
                    self.creds = flow.run_local_server(port=8080, open_browser=True)

                # Save tokens for next time
                async with aiofiles.open(self.token_file, "w") as token:
                    await token.write(self.creds.to_json())
                logger.info(f"💾 Tokens saved to {self.token_file}")

            # Build Gmail service
            self.service = build("gmail", "v1", credentials=self.creds)
            self._authenticated = True
            logger.info("✅ Gmail API ready!")
            return True

        except Exception as e:
            logger.error(f"❌ Gmail authentication failed: {e}")
            return False

    async def get_recent_emails(
        self, max_results: int = 10, query: str = "", time_filter: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Get recent emails with optional query filter
        Args:
            max_results: Maximum number of emails to fetch
            query: Gmail search query (e.g., 'from:noreply@example.com')
            time_filter: Time filter (e.g., '5m', '1h', '1d')
        Returns:
            List of email dictionaries with parsed content
        """
        if not self.is_authenticated():
            logger.error("❌ Gmail service not authenticated. Call authenticate() first.")
            return []

        try:
            # Add time filter to query if provided
            if time_filter and "newer_than:" not in query:
                query = f"newer_than:{time_filter} {query}".strip()

            logger.info(f"📧 Fetching {max_results} recent emails...")
            if query:
                logger.debug(f"🔍 Query: {query}")

            # Get message list
            assert self.service is not None
            results = self.service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()

            messages = results.get("messages", [])
            if not messages:
                logger.info("📭 No messages found")
                return []

            logger.info(f"📨 Found {len(messages)} messages, fetching details...")

            # Get full message details
            emails = []
            for i, message in enumerate(messages, 1):
                logger.debug(f"📖 Reading email {i}/{len(messages)}...")

                full_message = (
                    self.service.users().messages().get(userId="me", id=message["id"], format="full").execute()
                )

                email_data = self._parse_email(full_message)
                emails.append(email_data)

            return emails

        except HttpError as error:
            logger.error(f"❌ Gmail API error: {error}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching emails: {e}")
            return []

    def _parse_email(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gmail message into readable format"""
        headers = {h["name"]: h["value"] for h in message["payload"]["headers"]}

        return {
            "id": message["id"],
            "thread_id": message["threadId"],
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "timestamp": int(message["internalDate"]),
            "body": self._extract_body(message["payload"]),
            "raw_message": message,
        }

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from payload"""
        body = ""

        if payload.get("body", {}).get("data"):
            # Simple email body
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
        elif payload.get("parts"):
            # Multi-part email
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain" and part.get("body", {}).get("data"):
                    part_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    body += part_body
                elif part["mimeType"] == "text/html" and not body and part.get("body", {}).get("data"):
                    # Fallback to HTML if no plain text
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")

        return body


@register_toolkit("gmail")
class GmailToolkit(AsyncBaseToolkit):
    """
    Toolkit for Gmail integration.

    Provides functionality for:
    - Gmail API authentication via OAuth2 or access tokens
    - Reading recent emails with filtering and search
    - 2FA code extraction and verification
    - Email content analysis

    Required configuration:
    - GMAIL_ACCESS_TOKEN: Direct access token (optional, alternative to OAuth)
    - GMAIL_CREDENTIALS_FILE: Path to OAuth credentials JSON (optional)
    - GMAIL_TOKEN_FILE: Path to store/load access tokens (optional)

    Environment variables:
    - GMAIL_ACCESS_TOKEN: Direct Gmail access token
    - GMAIL_CREDENTIALS_PATH: Path to OAuth credentials file
    - GMAIL_TOKEN_PATH: Path to token storage file
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the GmailToolkit.

        Args:
            config: Toolkit configuration containing Gmail settings
        """
        super().__init__(config)

        # Get configuration from environment or config
        access_token = self.config.config.get("GMAIL_ACCESS_TOKEN") or os.getenv("GMAIL_ACCESS_TOKEN")
        credentials_file = self.config.config.get("GMAIL_CREDENTIALS_FILE") or os.getenv("GMAIL_CREDENTIALS_PATH")
        token_file = self.config.config.get("GMAIL_TOKEN_FILE") or os.getenv("GMAIL_TOKEN_PATH")

        # Initialize Gmail service
        self.gmail_service = GmailService(
            credentials_file=credentials_file, token_file=token_file, access_token=access_token
        )

        if not access_token and not credentials_file:
            self.logger.warning(
                "No Gmail access token or credentials file configured. "
                "Set GMAIL_ACCESS_TOKEN or GMAIL_CREDENTIALS_PATH environment variable "
                "or provide via ToolkitConfig"
            )

    async def authenticate_gmail(self) -> str:
        """
        Authenticate with Gmail API using OAuth2 or provided access token.

        Returns:
            Authentication status message
        """
        try:
            self.logger.info("Authenticating with Gmail API...")

            if await self.gmail_service.authenticate():
                return "✅ Successfully authenticated with Gmail API"
            else:
                return "❌ Failed to authenticate with Gmail API. Please check credentials."

        except Exception as e:
            error_msg = f"Gmail authentication error: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    async def get_recent_emails(self, keyword: str = "", max_results: int = 10, time_filter: str = "1h") -> str:
        """
        Get recent emails from the mailbox with optional keyword filtering.

        This tool is particularly useful for:
        - Retrieving verification codes, OTP, 2FA tokens from recent emails
        - Finding magic links for account verification
        - Reading recent email content for specific services
        - Monitoring for new messages from specific senders

        Args:
            keyword: A single keyword for search (e.g., 'github', 'airbnb', 'verification', 'otp')
            max_results: Maximum number of emails to retrieve (1-50, default: 10)
            time_filter: Time window for search ('5m', '1h', '1d', '1w', default: '1h')

        Returns:
            Formatted string with email details including subject, sender, date, and content
        """
        try:
            # Ensure authentication
            if not self.gmail_service.is_authenticated():
                self.logger.info("📧 Gmail not authenticated, attempting authentication...")
                if not await self.gmail_service.authenticate():
                    return "❌ Failed to authenticate with Gmail. Please ensure Gmail credentials are set up properly."

            # Validate parameters
            max_results = max(1, min(max_results, 50))  # Clamp between 1-50

            # Build query with time filter and optional keyword
            query_parts = [f"newer_than:{time_filter}"]
            if keyword.strip():
                query_parts.append(keyword.strip())

            query = " ".join(query_parts)
            self.logger.info(f"🔍 Gmail search query: {query}")

            # Get emails
            emails = await self.gmail_service.get_recent_emails(
                max_results=max_results, query=query, time_filter=time_filter
            )

            if not emails:
                query_info = f" matching '{keyword}'" if keyword.strip() else ""
                return f"📭 No recent emails found from last {time_filter}{query_info}"

            # Format results
            content = (
                f'📧 Found {len(emails)} recent email{"s" if len(emails) > 1 else ""} from the last {time_filter}:\n\n'
            )

            for i, email in enumerate(emails, 1):
                content += f"Email {i}:\n"
                content += f'From: {email["from"]}\n'
                content += f'Subject: {email["subject"]}\n'
                content += f'Date: {email["date"]}\n'
                content += f'Content:\n{email["body"]}\n'
                content += "-" * 50 + "\n\n"

            self.logger.info(f"📧 Retrieved {len(emails)} recent emails")
            return content

        except Exception as e:
            error_msg = f"Error getting recent emails: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    async def search_emails(self, query: str, max_results: int = 10, time_filter: Optional[str] = None) -> str:
        """
        Search emails using Gmail search syntax.

        Supports full Gmail search operators:
        - from:sender@example.com - emails from specific sender
        - to:recipient@example.com - emails to specific recipient
        - subject:keyword - emails with keyword in subject
        - has:attachment - emails with attachments
        - is:unread - unread emails only
        - newer_than:1d - emails newer than 1 day
        - older_than:1w - emails older than 1 week
        - label:inbox - emails in specific label

        Args:
            query: Gmail search query using Gmail search operators
            max_results: Maximum number of emails to retrieve (1-50, default: 10)
            time_filter: Optional time filter to add to query ('5m', '1h', '1d', '1w')

        Returns:
            Formatted string with matching email details

        Example queries:
            - "from:noreply@github.com" - All emails from GitHub
            - "subject:verification code" - Emails with "verification code" in subject
            - "from:security@google.com has:attachment" - Security emails from Google with attachments
        """
        try:
            # Ensure authentication
            if not self.gmail_service.is_authenticated():
                if not await self.gmail_service.authenticate():
                    return "❌ Failed to authenticate with Gmail. Please ensure Gmail credentials are set up properly."

            # Validate parameters
            max_results = max(1, min(max_results, 50))

            # Add time filter if provided
            full_query = query
            if time_filter and "newer_than:" not in query and "older_than:" not in query:
                full_query = f"newer_than:{time_filter} {query}".strip()

            self.logger.info(f"🔍 Gmail search query: {full_query}")

            # Search emails
            emails = await self.gmail_service.get_recent_emails(
                max_results=max_results,
                query=full_query,
                time_filter=time_filter or "30d",  # Default to 30 days if no time filter
            )

            if not emails:
                return f"📭 No emails found matching query: {full_query}"

            # Format results
            content = f'🔍 Found {len(emails)} email{"s" if len(emails) > 1 else ""} matching "{query}":\n\n'

            for i, email in enumerate(emails, 1):
                content += f"Email {i}:\n"
                content += f'From: {email["from"]}\n'
                content += f'Subject: {email["subject"]}\n'
                content += f'Date: {email["date"]}\n'
                content += f'Content:\n{email["body"][:500]}{"..." if len(email["body"]) > 500 else ""}\n'
                content += "-" * 50 + "\n\n"

            self.logger.info(f"🔍 Found {len(emails)} emails matching search")
            return content

        except Exception as e:
            error_msg = f"Error searching emails: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    async def get_verification_codes(self, sender_keyword: str = "", time_filter: str = "10m") -> str:
        """
        Extract verification codes, OTP tokens, and 2FA codes from recent emails.

        This tool specifically looks for common patterns in verification emails:
        - Numeric verification codes (4-8 digits)
        - Alphanumeric codes
        - Terms like "verification code", "OTP", "2FA", "authentication"
        - Magic links for verification

        Args:
            sender_keyword: Filter by sender keyword (e.g., 'google', 'github', 'apple')
            time_filter: Time window for search (default: '10m' for recent codes)

        Returns:
            Extracted verification codes and relevant email excerpts
        """
        try:
            # Build search query focused on verification emails
            query_parts = ["newer_than:" + time_filter]

            # Add verification-related keywords
            verification_keywords = [
                "verification",
                "OTP",
                "2FA",
                "authentication",
                "code",
                "verify",
                "confirm",
                "security",
                "sign in",
                "login",
            ]

            if sender_keyword.strip():
                query_parts.append(f"from:{sender_keyword}")

            # Add verification keywords to query
            keyword_query = " OR ".join(verification_keywords)
            query_parts.append(f"({keyword_query})")

            query = " ".join(query_parts)

            # Get emails
            emails = await self.gmail_service.get_recent_emails(
                max_results=20, query=query, time_filter=time_filter  # Check more emails for verification codes
            )

            if not emails:
                return f"📭 No verification emails found in the last {time_filter}"

            # Extract codes from emails
            codes_found = []

            import re

            # Patterns for common verification codes - more specific patterns
            code_patterns = [
                r"(?:code|otp|verification)[\s:]+(\d{4,8})",  # "code: 123456", "OTP: 123456"
                r"\b(\d{6})\b",  # Common 6-digit codes
                r"\b(\d{4,5})\b(?=\s*(?:is|to|for|$))",  # 4-5 digit codes followed by context
                r"(?:your|the)\s+(?:code|otp)[\s:]+(\d{4,8})",  # "your code: 123456"
                r"enter[\s\w]*?(\d{4,8})",  # "enter this code 123456"
            ]

            for email in emails:
                email_content = f"{email['subject']} {email['body']}"

                for pattern in code_patterns:
                    matches = re.findall(pattern, email_content, re.IGNORECASE)
                    for match in matches:
                        # Filter out obviously non-code numbers (years, common numbers)
                        if isinstance(match, str) and len(match) >= 4:
                            if not match in ["2023", "2024", "2025", "1234", "0000"]:
                                codes_found.append(
                                    {
                                        "code": match,
                                        "from": email["from"],
                                        "subject": email["subject"],
                                        "date": email["date"],
                                    }
                                )

            if not codes_found:
                return (
                    f"🔍 Found {len(emails)} verification emails but no codes could be extracted. Check emails manually."
                )

            # Format results
            content = (
                f"🔐 Found {len(codes_found)} potential verification code{'s' if len(codes_found) > 1 else ''}:\n\n"
            )

            for i, code_info in enumerate(codes_found[:5], 1):  # Show top 5 codes
                content += f"Code {i}: {code_info['code']}\n"
                content += f"From: {code_info['from']}\n"
                content += f"Subject: {code_info['subject']}\n"
                content += f"Date: {code_info['date']}\n"
                content += "-" * 30 + "\n\n"

            return content

        except Exception as e:
            error_msg = f"Error extracting verification codes: {str(e)}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "authenticate_gmail": self.authenticate_gmail,
            "get_recent_emails": self.get_recent_emails,
            "search_emails": self.search_emails,
            "get_verification_codes": self.get_verification_codes,
        }
