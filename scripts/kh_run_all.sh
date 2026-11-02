#!/usr/bin/env bash
# Reruns every analysis step downstream of the district sample (data steps s00-s03 and s10's inputs are unchanged).
# Usage: bash kh_run_all.sh [logdir]
PY="${PY:-$TEMP/tloda/venv/Scripts/python.exe}"
LOG="${1:-../outputs/logs}"; mkdir -p "$LOG"
cd "$(dirname "$0")"
for s in kh_s10_rwi kh_s04_models kh_s24_conservative_p kh_s25_joint_test kh_s27_need_conservative kh_s26_analysis_dataset kh_s07_shapley \
         kh_s18_wild_bootstrap kh_s15_access2017 kh_s06_ladder kh_s08_extensions kh_s09_geobias kh_s16_access_sensitivity kh_s17_sample_and_size \
         kh_s20_index_audit kh_s21_speed_iv kh_s22_timing_check kh_s23_usage_measures kh_s28_hospital_check kh_s29_external_validation \
         kh_main kh_figures kh_s11_webexport kh_s12_site_files kh_s05_maps; do
  t0=$(date +%s)
  if "$PY" -X utf8 -W ignore "$s.py" > "$LOG/$s.log" 2>&1; then st=ok; else st=FAILED; fi
  echo "$(date +%H:%M:%S) $s $st $(( $(date +%s) - t0 ))s"
done
