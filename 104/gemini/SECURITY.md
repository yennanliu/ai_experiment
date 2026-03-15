# 🔒 SECURITY WARNING

## ⚠️ CRITICAL: Credential Leak Detected and Fixed

This repository previously contained committed credentials in the following files:
- `104_auto_apply_complete.js` (email + password)
- `QUICK_REFERENCE.md` (email)
- `README_104_AUTOMATION.md` (email)
- `TOTAL_RESULTS.txt` (email + name)
- `CLAUDE.md` (email + password)

## ✅ Actions Taken

1. **Removed all hardcoded credentials** from tracked files
2. **Created `.env.example`** for secure configuration
3. **Updated `.gitignore`** to prevent future leaks
4. **Cleaned git history** to remove all traces of credentials
5. **Force pushed** to rewrite remote history

## 🔐 Secure Configuration

### Using Environment Variables

Create a `.env` file (NEVER commit this):

```bash
JOB_EMAIL=your_email@example.com
JOB_PASSWORD=your_secure_password
COVER_LETTER=自訂推薦信1
```

### Load in Your Script

```javascript
// Load environment variables (if using dotenv)
require('dotenv').config();

const email = process.env.JOB_EMAIL;
const password = process.env.JOB_PASSWORD;
```

## 🛡️ Security Best Practices

### NEVER Commit:
- ❌ Passwords or API keys
- ❌ Email addresses
- ❌ Personal information
- ❌ `.env` files
- ❌ Credentials in comments or documentation

### ALWAYS:
- ✅ Use environment variables
- ✅ Use `.env.example` templates
- ✅ Keep `.env` in `.gitignore`
- ✅ Rotate passwords after leaks
- ✅ Review commits before pushing

## 🔄 Password Rotation Required

**IMPORTANT:** If you were using the leaked credentials:
1. **Change your 104.com.tw password immediately**
2. **Enable 2FA** if not already enabled
3. **Review account activity** for unauthorized access
4. **Update password in your local `.env`** file

## 📋 .gitignore Entries

Make sure these are in your `.gitignore`:

```
# Credentials and secrets
.env
.env.local
.env.*.local
*.key
*.pem
credentials.json
config.local.js

# Personal information
*_credentials.txt
*_passwords.txt
```

## 🔍 How to Check for Leaks

```bash
# Search for potential credentials in current files
grep -r "password\|email.*@\|credentials" --exclude-dir=.git .

# Check git history for sensitive data
git log --all --full-history --source --pretty=format:'%h %s' | grep -i 'password\|secret\|credential'
```

## ⚡ Quick Reference

| Do This | Not This |
|---------|----------|
| `email: process.env.JOB_EMAIL` | `email: 'user@example.com'` |
| Store in `.env` file | Hardcode in scripts |
| Add `.env` to `.gitignore` | Commit `.env` file |
| Use `<placeholder>` in docs | Use real credentials in docs |

## 📞 If You Find a Leak

1. **Stop immediately** - Don't commit further
2. **Remove credentials** from files
3. **Clean git history** (see below)
4. **Rotate compromised credentials**
5. **Update security documentation**

## 🧹 Cleaning Git History

See the git history cleanup section in this repo for detailed instructions.

---

**Last Updated:** 2026-02-28
**Status:** Credentials removed, history cleaned, secure system in place
