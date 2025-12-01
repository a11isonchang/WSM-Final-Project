from sentence_transformers import SentenceTransformer

model_name = "intfloat/multilingual-e5-small"
save_dir = "models/multilingual-e5-small"  # 你自己決定資料夾位置

model = SentenceTransformer(model_name)
model.save(save_dir)

print("saved to", save_dir)

