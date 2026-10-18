#!/bin/sh
# Prints the torch build inside an inference image and whether it sees a GPU.
# usage: scripts/accel-check.sh <image> [gpu|cpu]
# With "gpu" (the default when the host has a GPU) it fails unless torch sees one.
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
	echo "usage: $0 <image> [gpu|cpu]" >&2
	exit 2
fi

image="$1"
expect=${2:-}
host_gpu=none

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
	host_gpu=nvidia
	set -- --gpus all
elif [ -e /dev/kfd ]; then
	host_gpu=amd
	# Group names such as "render" do not exist in slim images, so pass the host's numeric GIDs.
	set -- --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)"
	for node in /dev/dri/renderD*; do
		[ -e "$node" ] && set -- "$@" --group-add "$(stat -c %g "$node")"
	done
else
	set --
fi

if [ -z "$expect" ]; then
	if [ "$host_gpu" = none ]; then expect=cpu; else expect=gpu; fi
fi

out=$(docker run --rm "$@" "$image" python -c "import torch; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version, 'hip', None))")
printf '%s\n' "$out"

available=$(printf '%s\n' "$out" | tail -n 1 | awk '{print $2}')
if [ "$expect" = gpu ] && [ "$available" != "True" ]; then
	echo "expected a GPU ($host_gpu host) but torch.cuda.is_available() is $available" >&2
	exit 1
fi
