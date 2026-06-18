#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-/home/nguyenthanhlam/outputs/demo_stage1_stage2_db1_10}"
SEED="${SEED:-20260618}"
DDIM_STEPS="${DDIM_STEPS:-50}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-10}"

IMPOSE_ROOT="/home/nguyenthanhlam/IMPOSEStage1"
FINGEN_ROOT="/home/nguyenthanhlam/Fingen"

source /home/nguyenthanhlam/impose_env/bin/activate
export PYTHONPATH="${IMPOSE_ROOT}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python "${IMPOSE_ROOT}/jax_stage2/tools/generate_stage1_stage2_identity_dataset.py" \
  --stage1_config "${IMPOSE_ROOT}/jax_stage1/configs/rolled_ldm_vm_official_impose_probe.yaml" \
  --stage2_config "${IMPOSE_ROOT}/jax_stage2/configs/fvc2004_db1a_stage2_unet_lora_r8_5k_sample.yaml" \
  --stage1_ckpt /home/nguyenthanhlam/checkpoints/impose_official_stage1_unet_346632 \
  --controlnet_ckpt /home/nguyenthanhlam/checkpoints/core_exp0_self_condition_unetinit_bs256_50k/step_0050000 \
  --context_source none \
  --outdir "${OUTDIR}" \
  --sensors DB1A \
  --identity_sensors DB1A \
  --num_identities 1 \
  --instances_per_identity 10 \
  --stage1_batch_size 1 \
  --stage2_batch_size "${STAGE2_BATCH_SIZE}" \
  --stage1_ddim_steps "${DDIM_STEPS}" \
  --stage2_ddim_steps "${DDIM_STEPS}" \
  --ddim_eta 0.0 \
  --control_scale 1.0 \
  --seed "${SEED}" \
  --ridge_method sauvola \
  --ridge_window 11 \
  --ridge_k 0.007 \
  --reject_bad_identities \
  --pose_align_control \
  --pose_mask_image_dir DB1A=/home/nguyenthanhlam/data/fvc2004_db1a_png512 \
  --pose_masks_per_sensor 32 \
  --pose_mask_selection cycle \
  --pose_mask_mode intensity \
  --pose_align_transform bounded_similarity \
  --pose_scale_basis bbox_min \
  --pose_scale_min 0.75 \
  --pose_scale_max 1.25 \
  --pose_min_intersection_frac 0.01 \
  --save_pose_aligned_hints \
  --control_aug_enable \
  --control_aug_rotate_deg 2.0 \
  --control_aug_translate_frac 0.01 \
  --control_aug_dropout_frac 0.01 \
  --control_aug_speckle_frac 0.0005 \
  --overwrite

python "${FINGEN_ROOT}/scripts/make_demo_stage1_stage2_montage.py" "${OUTDIR}"

echo
echo "Demo output:"
echo "  ${OUTDIR}"
echo "Montage:"
echo "  ${OUTDIR}/demo_visuals/db1_stage1_stage2_demo.png"
