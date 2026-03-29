#!/bin/bash
set -e

echo "=== Gaussian Toolkit Container Starting ==="

# First-run: copy models from staging volume
if [ -d "/models-staging" ] && [ "$(ls -A /models-staging 2>/dev/null)" ]; then
    echo "Copying models from staging volume..."
    for subdir in /models-staging/*/; do
        dirname=$(basename "$subdir")
        target="/opt/models/$dirname"
        if [ ! -d "$target" ] || [ -z "$(ls -A "$target" 2>/dev/null)" ]; then
            mkdir -p "$target"
            cp -rn "$subdir"* "$target/" 2>/dev/null || true
            echo "  Copied $dirname"
        fi
    done
    echo "Model staging complete."
fi

# Link ComfyUI model directories
if [ -d "/opt/comfyui" ]; then
    for subdir in diffusion_models text_encoders vae loras checkpoints; do
        src="/opt/models/$subdir"
        dst="/opt/comfyui/models/$subdir"
        if [ -d "$src" ] && [ ! -L "$dst" ]; then
            rm -rf "$dst"
            ln -sf "$src" "$dst"
        fi
    done
    for subdir in trellis2 sam3d sam2 grounding-dino sams hunyuan3d UltraShape sam3; do
        src="/opt/models/$subdir"
        dst="/opt/comfyui/models/$subdir"
        if [ -d "$src" ] && [ ! -L "$dst" ]; then
            ln -sf "$src" "$dst"
        fi
    done
fi

# Start Xvfb (clean up stale locks from previous runs)
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 1
export DISPLAY=:1
fluxbox &>/dev/null &
x11vnc -display :1 -forever -nopw -shared -rfbport 5901 -bg 2>/dev/null || true

echo "VNC on port 5901"

# Ensure data directories exist
mkdir -p /data/output /data/input

# Create usd-assemble wrapper that uses the correct venv
cat > /usr/local/bin/usd-assemble << 'USDEOF'
#!/bin/bash
PYTHONPATH="" exec /opt/venv-usd/bin/python3 /opt/gaussian-toolkit/scripts/assemble_usd_scene.py "$@"
USDEOF
chmod +x /usr/local/bin/usd-assemble

echo "Services starting via supervisord..."

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/gaussian-toolkit.conf
