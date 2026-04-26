#!/bin/bash
# Script to check disk usage and find large directories

echo "=== Disk Usage Overview ==="
df -h /

echo ""
echo "=== Top 20 largest directories in / ==="
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20

echo ""
echo "=== Docker disk usage ==="
docker system df

echo ""
echo "=== Docker images (sorted by size) ==="
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}" | sort -k3 -hr

echo ""
echo "=== Docker containers (including stopped) ==="
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" 2>/dev/null || docker ps -a

echo ""
echo "=== Large files (>100MB) in /home/ec2-user ==="
find /home/ec2-user -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $9, $5}' | sort -k2 -hr | head -20
