#!/bin/bash

runtime_stats_log() {
  local message="$*"
  local timestamp
  timestamp="$(date -Is)"
  printf '[%s] %s\n' "${timestamp}" "${message}" | tee -a "${RUNTIME_STATS_SUMMARY}"
}

runtime_stats_init() {
  RUNTIME_STATS_OUTPUT_DIR="$1"
  RUNTIME_STATS_NAME="$2"

  mkdir -p "${RUNTIME_STATS_OUTPUT_DIR}"

  local stats_id
  stats_id="${SLURM_JOB_ID:-local-$(date +%Y%m%dT%H%M%S)}"
  RUNTIME_STATS_PREFIX="${RUNTIME_STATS_OUTPUT_DIR}/${RUNTIME_STATS_NAME}-${stats_id}"
  RUNTIME_STATS_SUMMARY="${RUNTIME_STATS_PREFIX}.stats.log"
  RUNTIME_STATS_TIME_LOG="${RUNTIME_STATS_PREFIX}.time.log"
  RUNTIME_STATS_GPU_LOG="${RUNTIME_STATS_PREFIX}.gpu.csv"
  RUNTIME_STATS_STARTED_EPOCH="$(date +%s)"
  RUNTIME_STATS_GPU_MONITOR_PID=""

  : > "${RUNTIME_STATS_SUMMARY}"

  runtime_stats_log "stats_file=${RUNTIME_STATS_SUMMARY}"
  runtime_stats_log "time_log=${RUNTIME_STATS_TIME_LOG}"
  runtime_stats_log "gpu_log=${RUNTIME_STATS_GPU_LOG}"
  runtime_stats_log "job_name=${SLURM_JOB_NAME:-local}"
  runtime_stats_log "job_id=${SLURM_JOB_ID:-local}"
  runtime_stats_log "submit_dir=${SLURM_SUBMIT_DIR:-local}"
  runtime_stats_log "host=$(hostname)"
  runtime_stats_log "user=${USER:-unknown}"
  runtime_stats_log "pwd=$(pwd -P)"
  runtime_stats_log "slurm_partition=${SLURM_JOB_PARTITION:-none}"
  runtime_stats_log "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-none}"
  runtime_stats_log "slurm_gpus=${SLURM_GPUS:-none}"
  runtime_stats_log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}"

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    {
      echo "===== scontrol show job ${SLURM_JOB_ID} at start ====="
      scontrol show job "${SLURM_JOB_ID}" || true
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi
}

runtime_stats_start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    runtime_stats_log "gpu_monitor=unavailable_nvidia_smi"
    return 0
  fi

  if ! nvidia-smi -L >/dev/null 2>&1; then
    runtime_stats_log "gpu_monitor=unavailable_no_visible_gpu"
    return 0
  fi

  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 60 \
    > "${RUNTIME_STATS_GPU_LOG}" 2>> "${RUNTIME_STATS_SUMMARY}" &
  RUNTIME_STATS_GPU_MONITOR_PID="$!"
  runtime_stats_log "gpu_monitor=started pid=${RUNTIME_STATS_GPU_MONITOR_PID}"
}

runtime_stats_stop_gpu_monitor() {
  if [[ -n "${RUNTIME_STATS_GPU_MONITOR_PID:-}" ]]; then
    kill "${RUNTIME_STATS_GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${RUNTIME_STATS_GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    runtime_stats_log "gpu_monitor=stopped pid=${RUNTIME_STATS_GPU_MONITOR_PID}"
  fi
}

runtime_stats_run() {
  runtime_stats_log "command=$*"
  local errexit_was_set=0
  if [[ "$-" == *e* ]]; then
    errexit_was_set=1
    set +e
  fi

  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "${RUNTIME_STATS_TIME_LOG}" "$@"
  else
    runtime_stats_log "warning=/usr/bin/time not available; running without verbose process stats"
    "$@"
  fi
  local status="$?"

  if [[ "${errexit_was_set}" -eq 1 ]]; then
    set -e
  fi

  runtime_stats_log "command_exit_code=${status}"
  return "${status}"
}

runtime_stats_finish() {
  local exit_code="$?"

  runtime_stats_stop_gpu_monitor

  local ended_epoch
  ended_epoch="$(date +%s)"
  runtime_stats_log "started_epoch=${RUNTIME_STATS_STARTED_EPOCH}"
  runtime_stats_log "ended_epoch=${ended_epoch}"
  runtime_stats_log "wall_seconds=$((ended_epoch - RUNTIME_STATS_STARTED_EPOCH))"
  runtime_stats_log "script_exit_code=${exit_code}"

  if [[ -f "${RUNTIME_STATS_TIME_LOG}" ]]; then
    {
      echo "===== /usr/bin/time -v ====="
      cat "${RUNTIME_STATS_TIME_LOG}"
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sstat >/dev/null 2>&1; then
    {
      echo "===== sstat snapshot near script exit ====="
      sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize 2>/dev/null || true
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sacct >/dev/null 2>&1; then
    {
      echo "===== sacct snapshot near script exit ====="
      sacct -j "${SLURM_JOB_ID}" \
        --format=JobID,JobName%32,Partition,Account,AllocCPUS,ReqMem,Elapsed,Timelimit,State,ExitCode,MaxRSS,AveRSS,MaxVMSize,ConsumedEnergyRaw,AllocTRES%100 \
        --parsable2 || true
      echo "NOTE: sacct fields can be more complete after the job has fully left RUNNING state."
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v seff >/dev/null 2>&1; then
    {
      echo "===== seff snapshot near script exit ====="
      seff "${SLURM_JOB_ID}" || true
      echo "NOTE: seff is most reliable after the job has completed."
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v eacct >/dev/null 2>&1; then
    {
      echo "===== Snellius EAR eacct snapshot near script exit ====="
      eacct -j "${SLURM_JOB_ID}" || eacct "${SLURM_JOB_ID}" || true
    } >> "${RUNTIME_STATS_SUMMARY}" 2>&1
  fi

  runtime_stats_log "post_job_stats_hint=sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,JobName,Elapsed,State,ExitCode,MaxRSS,AveRSS,ConsumedEnergyRaw,AllocTRES"
  runtime_stats_log "post_job_efficiency_hint=seff ${SLURM_JOB_ID:-<jobid>}"
  runtime_stats_log "post_job_energy_hint=eacct -j ${SLURM_JOB_ID:-<jobid>}"
  runtime_stats_log "snellius_budget_hint=budget-overview"
}
