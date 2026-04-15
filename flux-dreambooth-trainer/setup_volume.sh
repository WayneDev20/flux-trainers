#!/usr/bin/env bash
# ============================================================
# Vast.ai Network Volume Helper — INFORMATIONAL ONLY
#
# ⚠️  LIMITATION: Vast.ai volumes are machine-local.
#     A volume can only be attached to instances running on
#     the SAME physical machine.  In practice, the machines
#     that offer storage rarely also offer GPU compute, so
#     the volume ends up unusable.
#
# RECOMMENDED WORKFLOW (no volume needed):
#
#   1. Build the pre-baked Docker image once:
#        ./build_push.sh <dockerhub_user>
#
#   2. Train (weights downloaded fresh each run, ~5 min):
#        ./vastai_train.sh \
#            --images ./my_images.zip \
#            --docker_image <dockerhub_user>/flux-lora-trainer:latest
#
#   Total per run: ~2 min boot + ~5 min weights + training
#   (vs 35 min full setup on a bare CUDA image)
#
# ─────────────────────────────────────────────────────────────
# VOLUME PATH (if you want to try — only works when a GPU offer
# appears on the same machine as the volume):
#
#   # Find machines with BOTH volume AND gpu offers:
#   python3 -c "
#   import subprocess, json
#   gpu = {o['machine_id']: o for o in json.loads(
#       subprocess.check_output(['vastai','search','offers',
#           'num_gpus=1 disk_space>=20','--order','dph_total','--raw']))}
#   vol = {v.get('machine_id'): v for v in json.loads(
#       subprocess.check_output(['vastai','search','volumes',
#           'disk_space>=60','--storage','60','--raw']))}
#   common = set(gpu) & set(vol)
#   print('Machines with both:', len(common))
#   for m in common:
#       g = gpu[m]; print(g['id'], g['gpu_name'], g['dph_total'])
#   "
#
#   # If any machines found, create volume then attach:
#   OFFER_ID=<gpu_offer_id>
#   VOL_OFFER_ID=<vol_offer_id_on_same_machine>
#   vastai create volume $VOL_OFFER_ID --size 60 --name flux-weights
#   # Then use: vastai_train.sh --volume_id <vol_id> ...
# ============================================================
echo "See comments in this file for the volume workflow."
echo "Recommended: use the pre-baked Docker image instead."
echo "  ./build_push.sh <dockerhub_user>"
echo "  ./vastai_train.sh --images ./my_images.zip --docker_image <user>/flux-lora-trainer:latest"
