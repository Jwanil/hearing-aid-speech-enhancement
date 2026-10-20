# execution/phase5_unet_film/email_results.py
import os, sys, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

RECIPIENT = "jwanilmodi10@gmail.com"
SUBJECT = "[Hearing Aid ML] Phase 5A retraining required after verification audit"

BODY = """Dear Jwanil,

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

def main():
    print(f"Target Recipient: {RECIPIENT}")
    print(f"Subject: {SUBJECT}")
    
    app_pwd = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    sender = os.environ.get("GMAIL_SENDER", RECIPIENT).strip()

    if app_pwd:
        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = RECIPIENT
            msg["Subject"] = SUBJECT
            msg.attach(MIMEText(BODY, "plain"))
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender, app_pwd)
                server.send_message(msg)
            print("Successfully sent report email to", RECIPIENT)
            return
        except Exception as e:
            print(f"SMTP send attempt failed: {e}")
    else:
        print("[Notice] GMAIL_APP_PASSWORD not set in environment. Email body prepared below.")

    print("\n" + "=" * 70)
    print(BODY)
    print("=" * 70)

if __name__ == "__main__":
    main()
