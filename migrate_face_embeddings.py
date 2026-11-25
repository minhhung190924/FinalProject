"""Migrate legacy candidate face features to InsightFace buffalo_l embeddings.

Only face_1.jpg (the straight/frontal registration photo) is used by default.
Do not average the left/right pose photos: large head turns can reduce identity-match quality.
Run: python migrate_face_embeddings.py --dry-run
     python migrate_face_embeddings.py
     python migrate_face_embeddings.py --force
"""
import argparse, os, shutil, sqlite3
from datetime import datetime
import cv2
import db
import face_utils

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATASET_DIR=os.path.join(BASE_DIR,"dataset")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dry-run",action="store_true",help="Chỉ kiểm tra, không ghi database")
    parser.add_argument("--force",action="store_true",help="Tạo lại embedding kể cả khi đã là buffalo_l")
    parser.add_argument("--photo",default="face_1.jpg",help="Ảnh đăng ký nhìn thẳng dùng để migrate")
    args=parser.parse_args()
    db.init_db()
    candidates=db.get_all_candidates()
    if not candidates:
        print("Không có thí sinh trong database.")
        return
    if not args.dry_run:
        backup=os.path.join(BASE_DIR,f"exam_system_before_embedding_migration_{datetime.now():%Y%m%d_%H%M%S}.db")
        shutil.copy2(db.DB_PATH,backup)
        print(f"Đã backup database: {backup}")
    face_utils.init_face_recognition()
    ok=skip=fail=0
    for c in candidates:
        current=c.get("features") or []
        if not args.force and c.get("embedding_model")=="buffalo_l" and len(current)>=256:
            skip+=1; print(f"[SKIP] {c['full_name']}: đã có embedding buffalo_l"); continue
        photo=os.path.join(DATASET_DIR,c["full_name"],args.photo)
        if not os.path.isfile(photo):
            fail+=1; print(f"[FAIL] {c['full_name']}: không có {photo}"); continue
        img=cv2.imread(photo)
        if img is None:
            fail+=1; print(f"[FAIL] {c['full_name']}: không đọc được ảnh"); continue
        observations=face_utils.extract_face_observations(img)
        if len(observations)!=1:
            fail+=1; print(f"[FAIL] {c['full_name']}: cần đúng 1 khuôn mặt, nhận được {len(observations)}"); continue
        obs=observations[0]
        if not obs["quality_ok"]:
            fail+=1; print(f"[FAIL] {c['full_name']}: {obs['quality_message']} {obs['quality']}"); continue
        if args.dry_run:
            print(f"[CHECK] {c['full_name']}: OK, dim={len(obs['embedding'])}, quality={obs['quality']}")
        else:
            db.update_candidate_features(c["id"],obs["embedding"],"buffalo_l")
            print(f"[UPDATED] {c['full_name']}: dim={len(obs['embedding'])}")
        ok+=1
    print(f"\nKết quả: thành công={ok}, bỏ qua={skip}, lỗi={fail}")
    if fail:
        print("Các trường hợp lỗi cần đăng ký lại hoặc cung cấp ảnh face_1.jpg rõ, nhìn thẳng.")

if __name__=="__main__":
    main()
