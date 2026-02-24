ENV_NAME?=mmi-mzi

.PHONY: env preflight generate-debug evaluate-debug train-forward-debug train-inverse-debug inverse-debug full-debug

env:
	conda env create -f environment.yml || conda env update -n $(ENV_NAME) -f environment.yml

preflight:
	python mmi_mzi_project.py preflight

generate-debug:
	python mmi_mzi_project.py generate --stage debug --run-name debug_v1 --dry-run --yes

evaluate-debug:
	python mmi_mzi_project.py evaluate --run-dir runs/debug_v1

train-forward-debug:
	python mmi_mzi_project.py train-forward --run-dir runs/debug_v1 --epochs 10

train-inverse-debug:
	python mmi_mzi_project.py train-inverse --run-dir runs/debug_v1 --epochs 10 --K 4

inverse-debug:
	python mmi_mzi_project.py inverse-design --run-dir runs/debug_v1 --target-er 20 --target-bw 40 --target-il 1.0 --n-samples 256 --top-k 20

full-debug: generate-debug evaluate-debug train-forward-debug train-inverse-debug inverse-debug

