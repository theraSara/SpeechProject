# 1. Download the dataset
python run_baselines.py --download

# 2. Run with default settings (5-shot and 10-shot, 5 trials)
python run_baselines.py --gsc_root ./data/SpeechCommands/speech_commands_v0.02

# 3. Quick test (1 trial, 5-shot only)
python run_baselines.py --k_shots 5 --n_trials 1

# 4. Full experiment
python run_baselines.py --k_shots 5 10 --n_trials 10 --output_dir ./results/full_run