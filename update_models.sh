#!/bin/bash

# 🛠️ Config
BUCKET_NAME="pilltrack-mlops-storage"

# ✅ ชี้ไปที่ latest เลย (Script นี้จะเป็นอมตะ ไม่ต้องแก้เลขเวอร์ชันอีกแล้ว)
S3_SOURCE="s3://$BUCKET_NAME/releases/latest" 
LOCAL_DEST="./models"

echo "🚀 Starting Model Update from S3 (Production Latest)..."

# เช็คเน็ตก่อน (กันเหนียว)
if ! ping -c 1 google.com &> /dev/null; then
    echo "❌ No Internet Connection. Aborting."
    exit 1
fi

mkdir -p $LOCAL_DEST/pill
mkdir -p $LOCAL_DEST/box

# 1. Sync Pill 💊
echo "⬇️ Syncing Pill Models..."
aws s3 sync "$S3_SOURCE/pill" "$LOCAL_DEST/pill" --delete

# 2. Sync Box 📦
echo "⬇️ Syncing Box Models..."
aws s3 sync "$S3_SOURCE/box" "$LOCAL_DEST/box" --delete

echo "✅ Update Complete! Models are now at the latest version."
ls -R models/
