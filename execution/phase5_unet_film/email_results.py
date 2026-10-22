# execution/phase5_unet_film/email_results.py
import os, sys, smtplib, argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Helper to automatically load .env without requiring extra packages
def load_env():
    # Try python-dotenv if available
    try:
        from dotenv import load_dotenv
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env_path = os.path.join(root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
    except ImportError:
        pass

    # Fallback basic .env parser
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

load_env()

DEFAULT_RECIPIENT = "jwanilmodi10@gmail.com"
DEFAULT_SUBJECT = "[Hearing Aid ML] Phase 5A training / evaluation report"

DEFAULT_BODY = """Dear Jwanil,

The Phase 5A verification audit is complete. The historical training numbers
are internally consistent, but the old evaluation used the dynamic training
clean filelist and did not preserve enough provenance for final acceptance.

The Phase 5A workflow has been corrected to:
- evaluate official VoiceBank-DEMAND clean_testset_wav/noisy_testset_wav pairs;
- write a deterministic manifest and per-sample CSV;
- record checkpoint, manifest, CSV, Git, Python, PyTorch, CUDA, and GPU hashes;
- report Phase A as an offline denoising foundation, not as a causal or HASPI result;
- keep Phase B disabled until it has a validated HASPI surrogate and real discriminator.

Please run the exact commands in phase5a_retraining_prompt.md on the RTX 4090
workstation after connecting the SSD. Do not reuse the historical checkpoint
for final reporting; retrain from scratch with the corrected code.

Report path:
docs/phase_5a_verification_report.md
"""

def send_email(subject, body, recipient, sender=None, app_pwd=None):
    app_pwd = (app_pwd or os.environ.get("GMAIL_APP_PASSWORD", "")).replace(" ", "").strip()
    sender = (sender or os.environ.get("GMAIL_SENDER", recipient)).strip()

    if not app_pwd:
        print("\n[Action Required] GMAIL_APP_PASSWORD is not set.")
        print("To send automated emails via Gmail SMTP, follow these steps:")
        print("  1. Enable 2-Step Verification on your Google Account: https://myaccount.google.com/security")
        print("  2. Generate a 16-character App Password at: https://myaccount.google.com/apppasswords")
        print("     (Select app: 'Mail' or 'Other (Hearing Aid ML)')")
        print("  3. Create or update the .env file in the project root with:")
        print("     GMAIL_SENDER=jwanilmodi10@gmail.com")
        print("     GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
        print("  4. Or set environment variable in shell / PowerShell:")
        print("     $env:GMAIL_APP_PASSWORD=\"xxxx xxxx xxxx xxxx\"")
        print("\nPrepared email content:")
        print("=" * 70)
        print(f"To: {recipient}")
        print(f"Subject: {subject}")
        print("-" * 70)
        print(body)
        print("=" * 70)
        return False

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_pwd)
            server.send_message(msg)
        print(f"Successfully sent report email to {recipient} from {sender}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n[Error] Gmail SMTP Authentication Failed: {e}")
        print("Reason: Google rejected the username or app password.")
        print("Troubleshooting:")
        print("  - Ensure you are using a 16-character Google 'App Password', NOT your standard Gmail password.")
        print("  - Confirm 2-Step Verification is enabled on the sending account.")
        print("  - Verify that GMAIL_SENDER matches the Google account where the App Password was created.")
        return False
    except Exception as e:
        print(f"\n[Error] SMTP send attempt failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Send training/benchmark reports via Gmail SMTP")
    parser.add_argument("--recipient", default=DEFAULT_RECIPIENT, help="Recipient email address")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="Email subject line")
    parser.add_argument("--body", default=None, help="Inline email body text")
    parser.add_argument("--report", default=None, help="Path to markdown/text report file to send as body")
    args = parser.parse_args()

    body = DEFAULT_BODY
    if args.body:
        body = args.body
    elif args.report and os.path.exists(args.report):
        with open(args.report, "r", encoding="utf-8") as f:
            body = f.read()

    send_email(subject=args.subject, body=body, recipient=args.recipient)

if __name__ == "__main__":
    main()
