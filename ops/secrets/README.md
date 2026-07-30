# Sealed runtime configuration

Only ciphertext may be committed in this directory.

`lgi-readonly-database-url.enc` contains the LGI read-only PostgreSQL URL
encrypted with the production VPS public sealing key. The matching private key
never leaves `/opt/algonext-runtime-secrets` on the VPS. During deployment the
value is decrypted directly into `/opt/AlgoNext/.env` with mode `0600`.

The plaintext URL and PostgreSQL password must never be committed or printed in
CI logs.
