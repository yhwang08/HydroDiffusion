import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1)  collect only CSVs that contain “generic_diffusion” but *not* “flexschdlr”
# ------------------------------------------------------------------
stats_dir = "/home/yihan/diffusion_ssm/analysis/analysis/stats"
stats_dir = "/home/yihan/diffusion_ssm/analysis/analysis/ensemble_stats"

# per batch versus per epoch scheduler
# "generic_diffusion_ssm_flexscdlr_lr4e-4"
# "generic_diffusion_ssm_v2",
# "generic_diffusion_lstm_concatprcp",
# "generic_diffusion_lstm_perbatch",

# param grouping 
# "generic_diffusion_ssm_flexschlr_lr4e-4_allcond0_groupparams"
# "generic_diffusion_ssm_v2_flexscdlr_lr4e-4_allcond0_uniformgroups"

# conditioning of ssm
# "generic_diffusion_ssm_flexscdlr_lr4e-4"
# "generic_diffusion_ssm_v2_flexscdlr_lr4e-4_allcond0_uniformgroups"

# ssm encoder
# "generic_diffusion_ssm_unet_flexschlr_lr1e-4_fullcond_uniformparams",
# "generic_diffusion_ssm_lstm_flexschlr_lr4e-4_fullcond_groupparams"
# "generic_diffusion_ssm_flexschlr_lr1e-4_fullcond_uniformparams_ssmencoder_epoch58"
# "generic_diffusion_ssm_ssm_flexschlr_lr4e-4_cond0_paramgroups"
# "generic_diffusion_ssm_flexschlr_lr2e-4_cond0_embmlp_dropout0.2_seed5534"
# "generic_diffusion_ssm_lr2e-4_cond0_embmlp_linearreadout_dropout0.2_seed5534"  

LABEL = {
    #"generic_diffusion_ssm_unet_flexschlr_lr1e-4_fullcond_uniformparams": "ssm-unet",
    #"generic_diffusion_ssm_lstm_flexschlr_lr4e-4_fullcond_groupparams": "ssm-lstm",
    #"generic_diffusion_ssm_flexschlr_lr1e-4_fullcond_uniformparams_ssmencoder_epoch58": "ssm-ssm",
    #"generic_diffusion_ssm_ssm_flexschlr_lr4e-4_cond0_paramgroups": "ssm-ssm-2",
    #"generic_diffusion_lstm_perbatch": "lstm-lstm",
    #"generic_diffusion_lstm_flexschlr_dropout0.5": "lstm-lstm",
    #"generic_diffusion_ssm_flexschlr_lr2e-4_cond0_embmlp_dropout0.2_seed5534": "ssm-ssm-linear",
    #"generic_diffusion_ssm_lr2e-4_cond0_embmlp_dropout0.2_seed5534_encpower": "ssm-ssm-pp",
    #"diffusion_ssm_lr2e-4_cond0_silureadout_dropout0.2_seed5534_daymet_encpower": "ssm-ssm-pp-daymet",
    #"diffusion_ssm_v2_dmodel512_dstate512_bs64_daymet": "ssm-ssm-pp-daymet-d512-bs64",
    #"diffusion_ssm_v2_daymet_bs64": "ssm-ssm-pp-daymet-bs64",
    #"diffusion_ssm_v2_daymet_crossattention_concatprco": "ssm-ssm-crossatten-prcpconcat",
    #"diffusion_ssm_v2_xtallconcat_lion_seed5534": "ssm-ssm-xtallconcat-lion-seed5534",
    #"diffusion_ssm_v2_xtallconcat_lion_seed3407": "ssm-ssm-xtallconcat-lion-seed3407",
    #"diffusion_ssm_xtallconcat_pp_lion_seed3407_daymet": "ssm-ssm-daymet", # "ssm-ssm-xtallconcat-pp-lion-seed3407-daymet"
    #"diffusion_ssm_xtallconcat_lion_seed3407_daymet_bestepoch": "ssm-ssm-daymet-retrain-bestepoch", 
    
    #"diffusion_ssm_stallconcat_lion_seed3407_daymet_epoch60": "ssm-ssm-daymet-retrain-epoch60",
    #"diffusion_ssm_xtallconcat_lion_seed3407_all_bestepoch": "ssm-ssm-all-epoch60", 
    "diffusion_ssm_daymet_aligned_epoch60": "ssm-ssm-daymet-aligned-epoch60",
    #"diffusion_ssm_all_aligned_epoch60": "ssm-ssm-all-aligned-epoch60",
    
    #"diffusion_ssm_all_aligned_unnormy": "ssm-ssm-all-aligned-unnormy",
    #"diffusion_encoder_only_ssm_all_aligned_unnormy": "encoder-only-ssm-all-aligned-unnormy",
    
    "diffusion_encoder_only_ssm_daymet_aligned": "encoder-only-ssm-daymet-aligned",
    #"diffusion_encoder_only_ssm_all_aligned": "encoder-only-ssm-all-aligned",
    #"diffusion_ssm_daymet_aligned_epoch100": "ssm-ssm-daymet-aligned-epoch100",
    #"diffusdiffusion_ssm_xtallconcat_lion_seed3407_all_bestepoch": "ssm-ssm-all-bestepoch",
    
    #"diffusion_ssm_v2_daymet_bs64_540_8": "ssm-ssm-pp-daymet-bs64-540-8",
    
    #"diffusion_lstm_daymet_noise_cfg0": "lstm-lstm-daymet-noise",
    #"diffusion_lstm_daymet": "lstm-lstm-drum-setup",
    #"diffusion_lstm_xtallconcat_lion_seed3407":"lstm-lstm-all",
    #"diffusion_lstm_xtallconcat_lion_seed3407_daymet": "lstm-lstm-daymet", # "lstm-lstm-xtallconcat-lion-seed3407-daymet"
    "diffusion_lstm_daymet_aligned": "lstm-lstm-daymet-aligned",
    #"diffusion_lstm_aligned_all": "lstm-lstm-all-aligned",
    "diffusion_encoder_only_lstm_daymet_aligned_faster_det": "encoder-only-lstm-daymet-aligned-faster",
    "diffusion_encoder_only_lstm_daymet_aligned_det": "encoder-only-lstm-daymet-aligned",

    #"encdec_lstm_dropout0.5": "det encdec_lstm"
    #"encdec_lstm_seed3407_allforcings":"det encdec lstm (3-source forcings)",
    #"encdec_lstm_seed3407_daymet": "det-encdec-lstm-daymet",
    "encdec_lstm_daymet_aligned": "det-encdec-lstm-daymet-aligned",
    
    #"generic_diffusion_unet_attention_film_v2_h64": "lstm-unet",
    #"generic_diffusion_ssm_flexschlr_lr4e-4_allcond0_groupparams": "lstm-ssm",
}

LABEL = {
    "decoder_only_ssm_ddim20_det":"ddim20",
    "decoder_only_ssm_ddim15_det":"ddim15",
    "decoder_only_ssm_ddim5_det":"ddim5",
    "decoder_only_ssm_ddim3_det":"ddim3",
    "decoder_only_ssm_ddim2_det":"ddim2",
    #"decoder_only_ssm_ddim1_det":"ddim1",
    "diffusion_decoder_only_ssm_daymet_aligned_det":"ddim10",

}

wanted = set(LABEL)                      # the long names we keep

# ────────────────────────────────────────────────────────────────
# 2)  collect the files we care about
# ────────────────────────────────────────────────────────────────
file_list = []
for fp in glob.glob(os.path.join(stats_dir, "*.csv")):
    stem = Path(fp).stem.lower()          # no extension
    stem = re.sub(r"_lead\d+$", "", stem) # drop “…_leadXX”
    if stem in wanted:
        file_list.append(fp)

# ────────────────────────────────────────────────────────────────
# 3)  group files by model-stem and sort by lead time
# ────────────────────────────────────────────────────────────────
model_files  = {}
lead_re = re.compile(r'lead(\d+)')

for f in file_list:
    base  = os.path.basename(f)
    model = re.sub(r'_lead\d+\.csv$', "", base.lower())   # same stem as above
    lead  = int(lead_re.search(base).group(1))
    model_files.setdefault(model, []).append((lead, f))

for m in model_files:
    model_files[m].sort(key=lambda t: t[0])               # sort by lead

# ────────────────────────────────────────────────────────────────
# 4)  plotting — use LABEL[model] for the legend text
# ────────────────────────────────────────────────────────────────
metric_keys   = ['nse', 'kge', 'kge_r']
metric_titles = ['Median NSE', 'Median KGE', 'Median Correlation']

STYLE = {
    'ssm' :  dict(linestyle='-',  marker='s', markersize=6),
    'unet': dict(linestyle='--', marker='^', markersize=6),
    'lstm': dict(linestyle='-.', marker='o', markersize=6),
}
family = lambda name: 'ssm' if 'ssm' in name else ('unet' if 'unet' in name else 'lstm')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

for ax, metric, title in zip(axes, metric_keys, metric_titles):
    all_leads = set()

    for m_stem, files in model_files.items():
        leads, vals = [], []
        for lead, f in files:
            df = pd.read_csv(f, sep=None, engine="python")
            median = df.loc[df['basin'].str.lower() == 'median'].iloc[0]
            leads.append(lead-1)
            vals.append(float(median[metric]))
            all_leads.add(lead-1)

        ax.plot(
            leads,
            vals,
            label=LABEL[m_stem],         # ← short label instead of full name
            **STYLE[family(m_stem)]
        )

    ax.set_title(title)
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel("Performance")
    #ax.set_ylim([0.7, 0.9])
    ax.set_xticks(sorted(all_leads))
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("skill_with_leadtime_ddimtest.png", dpi=600)
plt.close()

