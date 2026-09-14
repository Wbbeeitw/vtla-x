import json, os, shutil, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file, safe_open

TOKEN = os.environ["HF_TOKEN"]  # read from env; never commit tokens
KW = dict(repo_id="google/paligemma-3b-pt-224", token=TOKEN,
          endpoint="https://huggingface.co",
          proxies={"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
EMB_RLINF = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

idx = json.load(open(hf_hub_download(filename="model.safetensors.index.json", **KW)))
embeds = {k: v for k, v in idx["weight_map"].items() if "embed_tokens" in k}
print("embed keys in repo:", embeds, flush=True)
target_key = [k for k in embeds if "language_model" in k][0]
print("downloading shard:", embeds[target_key], flush=True)
shard = hf_hub_download(filename=embeds[target_key], **KW)
print("shard cached at:", shard, flush=True)

with safe_open(shard, framework="pt", device="cpu") as f:
    emb = f.get_tensor(target_key)
print("embed shape:", tuple(emb.shape), "std:", round(emb.float().std().item(), 4), flush=True)

snap = "/data_vtla/hf/hub/models--lerobot--pi05_libero_base/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd"
sd = load_file(snap + "/model.safetensors")
sd[EMB_RLINF] = emb.to(sd[sorted(sd.keys())[0]].dtype)
out = "/data_vtlax/checkpoints/pi05_libero_base_openpi"
os.makedirs(out, exist_ok=True)
save_file(sd, out + "/model.safetensors", metadata={"format": "pt"})
shutil.copy(snap + "/config.json", out + "/config.json")
print("MERGED ->", out, flush=True)
