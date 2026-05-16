# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from safetensors.torch import load_file
from datetime import timedelta
from types import SimpleNamespace
import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens
from modeling.lladao import (
    LLaDAOConfig,
    LLaDAO,
    LLaDAConfig, 
    LLaDAModel,
    LLaDAModelLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers import AutoTokenizer, DINOv3ViTModel
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.lladao.llada_navit import NaiveCache


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    custom_timeout = timedelta(minutes=120) # 2 hours
    dist.init_process_group(backend="nccl", timeout=custom_timeout)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_model_weights(model, model_path, verbose=False):
    single_checkpoint_path = os.path.join(model_path, "ema.safetensors")
    sharded_index_path = os.path.join(model_path, "ema.safetensors.index.json")

    if os.path.exists(single_checkpoint_path):
        if verbose:
            print(f"[rank 0] Loading single checkpoint: {single_checkpoint_path}", flush=True)
        model_state_dict = load_file(single_checkpoint_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        del model_state_dict
        return msg

    if os.path.exists(sharded_index_path):
        if verbose:
            print(f"[rank 0] Loading sharded checkpoint index: {sharded_index_path}", flush=True)
        with open(sharded_index_path, "r", encoding="utf-8") as fp:
            index_data = json.load(fp)

        checkpoint_keys = set(index_data["weight_map"].keys())
        model_keys = set(model.state_dict().keys())
        missing_keys = sorted(model_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - model_keys)
        shard_names = sorted(set(index_data["weight_map"].values()))

        if verbose:
            print(f"[rank 0] Found {len(shard_names)} checkpoint shards", flush=True)

        for shard_idx, shard_name in enumerate(shard_names, start=1):
            shard_path = os.path.join(model_path, shard_name)
            if verbose:
                print(f"[rank 0] Loading shard {shard_idx}/{len(shard_names)}: {shard_name}", flush=True)
            shard_state_dict = load_file(shard_path, device="cpu")
            incompatible_keys = model.load_state_dict(shard_state_dict, strict=False)
            unexpected_keys.extend(incompatible_keys.unexpected_keys)
            del shard_state_dict

        return SimpleNamespace(
            missing_keys=missing_keys,
            unexpected_keys=sorted(set(unexpected_keys)),
        )

    raise FileNotFoundError(
        f"Neither ema.safetensors nor ema.safetensors.index.json was found under {model_path}"
    )


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None):
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)] * num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using the LLaDA-o model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default=os.environ.get("LLADAO_MODEL_PATH", ""))
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'], help='Model dtype')
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--dpg_bench', action='store_true', help='Use DPG bench prompt format (filename + prompt).')
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Please pass --model-path or set LLADAO_MODEL_PATH before running Geneval generation.")

    # Set dtype.
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")
        print(f"Using dtype: {args.dtype}")
        print("[rank 0] Loading llm config...", flush=True)

    llm_config = LLaDAConfig.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "LLaDAMoTDecoderLayer"

    if rank == 0:
        print("[rank 0] Loading vision config...", flush=True)
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    if rank == 0:
        print("[rank 0] Loading autoencoder...", flush=True)
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))
    repa_model = None
    if args.reg:
        if rank == 0:
            print("[rank 0] Loading REPA model...", flush=True)
        repa_model_path = os.environ.get("LLADAO_REPA_MODEL_PATH", "")
        if not repa_model_path:
            raise ValueError("Please set LLADAO_REPA_MODEL_PATH when running Geneval generation with --reg.")
        repa_model = DINOv3ViTModel.from_pretrained(repa_model_path)
    if rank == 0:
        print("[rank 0] Building LLaDA-o model...", flush=True)
    config = LLaDAOConfig(
        visual_gen=True,
        visual_und=True,
        visual_gen_repa=args.reg,
        visual_gen_reg=args.reg,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size, # 64 is for 1024x1024 resolutions
    )
    language_model = LLaDAModelLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    if args.reg:
        model = LLaDAO(language_model, vit_model, repa_model, config)
    else:
        model = LLaDAO(language_model, vit_model, None, config)
    if config.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    if rank == 0:
        print("[rank 0] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    if rank == 0:
        print("[rank 0] Loading model weights...", flush=True)
    msg = load_model_weights(model, args.model_path, verbose=(rank == 0))
    if rank == 0:
        print(msg)
        print("[rank 0] Moving model to device...", flush=True)

    model = model.to(device).to(model_dtype).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    if rank == 0:
        print("[rank 0] Model ready. Loading prompt metadata...", flush=True)

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    if args.resolution == 512:
        timestep_shift = 1.0
    elif args.resolution == 1024:
        timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total_metadatas = len(metadatas)
    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        metadata = metadatas[idx]
        folder_name = str(metadata.get("filename", f"{idx:0>5}")) if args.dpg_bench else f"{idx:0>5}"
        outpath = os.path.join(output_dir, folder_name)
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start} ({folder_name}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        flag = True
        for sample_idx in range(args.num_images):
            if not os.path.exists(os.path.join(sample_path, f"{sample_idx:05}.png")):
                flag = False
                break
        if flag:
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue

        if not args.dpg_bench:
            with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
                json.dump(metadata, fp)

        image_list = []

        remaining = args.num_images
        while remaining > 0:
            current_batch_size = min(args.batch_size, remaining)
            tmp_image_list = generate_image(
                prompt=prompt,
                cfg_scale=cfg_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                num_images=current_batch_size,
                resolution=args.resolution,
                device=device,
            )
            image_list.extend(tmp_image_list)
            remaining -= current_batch_size

        sample_count = 0
        for sample in image_list:
            bbox = sample.getbbox()
            if bbox is not None:
                sample = sample.crop(bbox)
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()
