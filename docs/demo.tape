# asr-attack demo recording, ~20 s GIF.
#
# Render with:  vhs docs/demo.tape
# Outputs:      docs/demo.gif
#
# Prereqs:
#   - VHS installed:           https://github.com/charmbracelet/vhs
#   - asr-attack venv ready:   uv sync
#   - HF cache warm (the demo uses ~5 s of benchmark time):
#       uv run python -c "from asr_attack import HFASRModel; \
#                         HFASRModel.from_pretrained('facebook/wav2vec2-base-960h', device='cpu')"

Output docs/demo.gif

Set FontSize 14
Set Width 1100
Set Height 640
Set Theme "Dracula"
Set TypingSpeed 25ms
Set Shell "bash"

Type "uv run python"
Sleep 300ms
Enter
Sleep 500ms

Type "from asr_attack import Attack, run_benchmark"
Enter
Sleep 3.2s

Type "report = run_benchmark("
Enter
Type "    model='facebook/wav2vec2-base-960h',"
Enter
Type "    attack=Attack.fgsm(epsilon=0.02),"
Enter
Type "    dataset='hf-internal-testing/librispeech_asr_dummy',"
Enter
Type "    n_samples=3, split='validation', config='clean',"
Enter
Type ")"
Enter
Sleep 13.5s

Type "print(report.summary())"
Enter
Sleep 3.5s

Type "report.to_html('report.html')"
Enter
Sleep 1.5s
