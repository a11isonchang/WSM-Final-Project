python - <<'EOF'
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
save_dir = "My_RAG/models/all_minilm_l6"
mod:qui.save(save_dir)

print("Saved files:", os.listdir(save_dir))
EOF
