# Example Usage

## Canny ControlNet V3

```bash
python3 main.py \
 --prompt "cyberpank dining room, full hd, cinematic" \
 --image input_canny1.png --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4
```