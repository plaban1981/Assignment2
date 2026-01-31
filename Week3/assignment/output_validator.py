"""
Output Validator - Validates LLM outputs for security violations.
Final defense against PII leakage, system exposure, and policy violations.
Modified for Internal Ops Desk context.
"""

import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class OutputValidator:
    """Validates LLM outputs before returning to users."""

    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'

    # Policy-sensitive action patterns (Internal Ops specific)
    POLICY_SENSITIVE_PATTERNS = [
        r'payroll\s+(change|modification|adjust)',
        r'salary\s+(change|modification|increase|decrease)',
        r'badge\s+(override|bypass|master)',
        r'access\s+(override|bypass|all\s+areas)',
        r'termination\s+(process|initiate)',
        r'disciplinary\s+action',
        r'confidential\s+employee\s+record',
    ]

    def __init__(self, allowed_emails: List[str] = None):
        """
        Initialize validator.

        Args:
            allowed_emails: List of email addresses that are safe to include
        """
        self.allowed_emails = allowed_emails or []

    def check_for_pii_leakage(self, text: str, user_email: str) -> Tuple[bool, List[str]]:
        """
        Check if output contains PII from other users.

        Args:
            text: Output text to check
            user_email: Current user's email (allowed)

        Returns:
            Tuple of (has_pii, violations)
        """
        violations = []

        # Check for emails
        emails = re.findall(self.EMAIL_PATTERN, text)
        for email in emails:
            if email != user_email and email not in self.allowed_emails:
                violations.append(f"Unauthorized email: {email}")

        # Check for phone numbers
        phones = re.findall(self.PHONE_PATTERN, text)
        if phones:
            violations.append(f"Phone number found: {phones[0]}")

        # Check for SSN patterns
        ssns = re.findall(self.SSN_PATTERN, text)
        if ssns:
            violations.append("SSN pattern found")

        return len(violations) > 0, violations

    def check_for_system_exposure(self, text: str) -> bool:
        """
        Check if output exposes system internals.

        Args:
            text: Output text to check

        Returns:
            True if system exposure detected
        """
        exposure_patterns = [
            r"system prompt",
            r"instruction:",
            r"<system>",
            r"```yaml",  # Might be leaking prompt files
            r"SECURITY CONSTRAINTS",
            r"Layer \d+",
        ]

        text_lower = text.lower()
        for pattern in exposure_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def check_policy_violations(
        self,
        action: str,
        response_text: str,
        requires_approval: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if action violates Internal Ops policies.

        Args:
            action: Action being taken
            response_text: Response text to check
            requires_approval: Whether approval is required

        Returns:
            Tuple of (is_violation, reason)
        """
        text_lower = response_text.lower()

        # Check for policy-sensitive content without approval
        for pattern in self.POLICY_SENSITIVE_PATTERNS:
            if re.search(pattern, text_lower):
                if not requires_approval:
                    return True, f"Policy-sensitive action detected without approval flag: {pattern}"

        # Specific action checks
        sensitive_actions = [
            'modify_payroll',
            'override_badge',
            'grant_admin_access',
            'access_personnel_records',
            'initiate_termination'
        ]

        if action in sensitive_actions and not requires_approval:
            return True, f"Action '{action}' requires approval"

        return False, None

    def validate(
        self,
        response_text: str,
        user_email: str,
        action: Optional[str] = None,
        requires_approval: Optional[bool] = None,
        ticket_type: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate output before returning to user.

        Args:
            response_text: The message text to validate
            user_email: Current user's email
            action: Action being taken (for policy checks)
            requires_approval: Whether approval is required
            ticket_type: Type of ticket for context-specific validation

        Returns:
            Tuple of (is_valid, error_type)
        """
        # Check for PII leakage
        has_pii, violations = self.check_for_pii_leakage(response_text, user_email)
        if has_pii:
            logger.error(f"PII leakage detected: {violations}")
            return False, "output_contains_pii"

        # Check for system exposure
        if self.check_for_system_exposure(response_text):
            logger.error("System information exposure detected")
            return False, "system_exposure"

        # Check policy violations (Internal Ops specific)
        if action:
            is_violation, reason = self.check_policy_violations(
                action, response_text, requires_approval or False
            )
            if is_violation:
                logger.error(f"Policy violation: {reason}")
                return False, "policy_violation"

        # All checks passed
        return True, None
