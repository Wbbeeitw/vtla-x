import torch
from safetensors.torch import load_file, save_file, safe_open
import glob

EMB = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
snap = "/data_vtla/hf/hub/models--lerobot--pi05_libero_base/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd"
out = "/data_vtlax/checkpoints/pi05_libero_base_openpi"

# locate the cached paligemma shard (already downloaded)
roots = [
    "/workspace/hf_home/hub/models--google--paligemma-3b-pt-224/snapshots/*/model-00001-of-00003.safetensors",
    "/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/*/model-00001-of-00003.safetensors",
    "/data_vtlax/hf_home/hub/models--google--paligemma-3b-pt-224/snapshots/*/model-00001-of-00003.safetensors",
]
shards = [s for r in roots for s in glob.glob(r)]
assert shards, "cached shard not found"
shard = shards[0]

# load current merged (16.5G, embed has 257216 rows) and the fresh pretrained embed
sd = load_file(out + "/model.safetensors")
with safe_open(shard, framework="pt", device="cpu") as f:
    emb = f.get_tensor("language_model.model.embed_tokens.weight")
print("repo embed:", tuple(emb.shape))
torch.save(emb.clone(), "/data_vtlax/paligemma_embed.pt")  # persist for future use

# slice off the 64-row vocab padding, match fp32
emb = emb[:257152].to(sd[sorted(sd.keys())[0]].dtype)
print("sliced embed:", tuple(emb.shape), "std:", round(emb.float().std().item(), 4))
sd[EMB] = emb
save_file(sd, out + "/model.safetensors", metadata={"format": "pt"})
print("resaved:", out, flush=True)

# strict verification against the real model
import sys
sys.path.insert(0, "/workspace/RLinf")
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config, OpenPi0ForRLActionPrediction
cfg = get_openpi_config("pi05_libero", model_path=out)
mc = OpenPi0Config(**cfg.model.__dict__)
model = OpenPi0ForRLActionPrediction(mc)
r = model.load_state_dict(load_file(out + "/model.safetensors"), strict=True)
print("STRICT LOAD OK — missing:", r.missing_keys, "unexpected:", r.unexpected_keys, flush=True)
e = model.state_dict()[EMB]
print("model embed now:", tuple(e.shape), "std:", round(e.std().item(), 4), flush=True)
