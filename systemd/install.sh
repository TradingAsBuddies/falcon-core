#!/bin/bash
# Install Falcon Feedback Loop as a systemd service
# Run as root or with sudo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALCON_USER="${FALCON_USER:-falcon}"
FALCON_HOME="${FALCON_HOME:-/opt/falcon}"

echo "=== Falcon Feedback Loop Service Installation ==="

# Create falcon user if it doesn't exist
if ! id "$FALCON_USER" &>/dev/null; then
    echo "Creating user: $FALCON_USER"
    useradd -r -m -s /bin/bash "$FALCON_USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p /etc/falcon
mkdir -p /var/lib/falcon
mkdir -p "$FALCON_HOME"

# Set up virtual environment if needed
if [ ! -d "$FALCON_HOME/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$FALCON_HOME/venv"
fi

# Install falcon-core
echo "Installing falcon-core..."
"$FALCON_HOME/venv/bin/pip" install --upgrade pip
"$FALCON_HOME/venv/bin/pip" install falcon-core[backtesting]

# Or install from local source if available
if [ -d "$(dirname "$SCRIPT_DIR")/src" ]; then
    echo "Installing from local source..."
    "$FALCON_HOME/venv/bin/pip" install -e "$(dirname "$SCRIPT_DIR")[backtesting]"
fi

# Install schedule library (required for daemon mode)
"$FALCON_HOME/venv/bin/pip" install schedule

# Copy environment file template
if [ ! -f /etc/falcon/environment ]; then
    echo "Creating environment file template..."
    cp "$SCRIPT_DIR/environment.example" /etc/falcon/environment
    chmod 600 /etc/falcon/environment
    echo "IMPORTANT: Edit /etc/falcon/environment with your API keys!"
fi

# Set ownership
chown -R "$FALCON_USER:$FALCON_USER" "$FALCON_HOME"
chown -R "$FALCON_USER:$FALCON_USER" /var/lib/falcon
chown "$FALCON_USER:$FALCON_USER" /etc/falcon/environment

# Install systemd service
echo "Installing systemd service..."
cp "$SCRIPT_DIR/falcon-feedback-loop.service" /etc/systemd/system/

# Update service file with correct user/home
sed -i "s|User=falcon|User=$FALCON_USER|g" /etc/systemd/system/falcon-feedback-loop.service
sed -i "s|Group=falcon|Group=$FALCON_USER|g" /etc/systemd/system/falcon-feedback-loop.service
sed -i "s|/opt/falcon|$FALCON_HOME|g" /etc/systemd/system/falcon-feedback-loop.service
sed -i "s|/home/falcon|/home/$FALCON_USER|g" /etc/systemd/system/falcon-feedback-loop.service

# Reload systemd
systemctl daemon-reload

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit /etc/falcon/environment with your API keys"
echo "2. Enable the service:  systemctl enable falcon-feedback-loop"
echo "3. Start the service:   systemctl start falcon-feedback-loop"
echo "4. Check status:        systemctl status falcon-feedback-loop"
echo "5. View logs:           journalctl -u falcon-feedback-loop -f"
echo ""
echo "To run a one-time test:"
echo "  sudo -u $FALCON_USER $FALCON_HOME/venv/bin/python -m falcon_core.backtesting.scheduler --run-now"
