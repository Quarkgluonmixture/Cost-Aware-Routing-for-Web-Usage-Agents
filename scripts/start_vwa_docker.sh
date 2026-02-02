#!/bin/bash
# VWA Docker Startup Script
# This script starts all required Docker containers for VWA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VWA_DIR="$PROJECT_DIR/external/visualwebarena"
ENV_DIR="$VWA_DIR/environment_docker"

echo "=== VWA Docker Startup Script ==="
echo "Project directory: $PROJECT_DIR"
echo "VWA directory: $VWA_DIR"
echo "Environment directory: $ENV_DIR"

# Check if required files exist
check_required_files() {
    echo ""
    echo "Checking required files..."

    # Check for shopping image
    if ! docker images | grep -q "shopping"; then
        echo "  [MISSING] Shopping Docker image (shopping_final_0712.tar)"
        echo "    Download from: https://drive.google.com/file/d/1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA/view?usp=sharing"
        MISSING_FILES=true
    else
        echo "  [OK] Shopping Docker image found"
    fi

    # Check for forum image
    if ! docker images | grep -q "postmill"; then
        echo "  [MISSING] Forum Docker image (postmill-populated-exposed-withimg.tar)"
        echo "    Download from: https://drive.google.com/file/d/17Qpp1iu_mPqzgO_73Z9BnFjHrzmX9DGf/view?usp=sharing"
        MISSING_FILES=true
    else
        echo "  [OK] Forum Docker image found"
    fi

    # Check for Wikipedia zim file
    if [ ! -f "$ENV_DIR/data/wikipedia_en_all_maxi_2022-05.zim" ]; then
        echo "  [MISSING] Wikipedia zim file"
        echo "    Download from: https://drive.google.com/file/d/1Um4QLxi_bGv5bP6kt83Ke0lNjuV9Tm0P/view?usp=sharing"
        echo "    Place in: $ENV_DIR/data/"
        MISSING_FILES=true
    else
        echo "  [OK] Wikipedia zim file found"
    fi

    # Check for classifieds
    if [ ! -d "$ENV_DIR/classifieds_docker_compose" ]; then
        echo "  [MISSING] Classifieds Docker Compose directory"
        echo "    Download from: https://drive.google.com/file/d/1m79lp84yXfqdTBHr6IS7_1KkL4sDSemR/view"
        echo "    Extract to: $ENV_DIR/"
        MISSING_FILES=true
    else
        echo "  [OK] Classifieds directory found"
    fi

    if [ "$MISSING_FILES" = true ]; then
        echo ""
        echo "ERROR: Some required files are missing. Please download them first."
        exit 1
    fi
}

# Start Shopping site
start_shopping() {
    echo ""
    echo "=== Starting Shopping Site (port 7770) ==="

    if docker ps | grep -q "shopping"; then
        echo "Shopping container is already running."
    else
        docker start shopping 2>/dev/null || docker run --name shopping -p 7770:80 -d shopping_final_0712
        echo "Waiting for shopping to start..."
        sleep 30

        # Configure shopping
        echo "Configuring shopping..."
        docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" 2>/dev/null || true
        docker exec shopping mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";' 2>/dev/null || true
        docker exec shopping /var/www/magento2/bin/magento cache:flush 2>/dev/null || true

        # Disable re-indexing
        for indexer in catalogrule_product catalogrule_rule catalogsearch_fulltext catalog_category_product customer_grid design_config_grid inventory catalog_product_category catalog_product_attribute catalog_product_price cataloginventory_stock; do
            docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule $indexer 2>/dev/null || true
        done

        echo "Shopping site started at http://localhost:7770"
    fi
}

# Start Reddit/Forum site
start_reddit() {
    echo ""
    echo "=== Starting Reddit Site (port 9999) ==="

    if docker ps | grep -q "forum"; then
        echo "Forum container is already running."
    else
        docker start forum 2>/dev/null || docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
        echo "Forum site started at http://localhost:9999"
    fi
}

# Start Wikipedia site
start_wikipedia() {
    echo ""
    echo "=== Starting Wikipedia Site (port 8888) ==="

    if docker ps | grep -q "wikipedia"; then
        echo "Wikipedia container is already running."
    else
        docker run -d --name=wikipedia --volume="$ENV_DIR/data/:/data" -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
        echo "Wikipedia site started at http://localhost:8888"
    fi
}

# Start Classifieds site
start_classifieds() {
    echo ""
    echo "=== Starting Classifieds Site (port 9980) ==="

    if docker ps | grep -q "classifieds"; then
        echo "Classifieds containers are already running."
    else
        cd "$ENV_DIR/classifieds_docker_compose"
        # Update docker-compose.yml with localhost URL
        sed -i "s|<your-server-hostname>|localhost|g" docker-compose.yml
        docker compose up --build -d
        cd "$PROJECT_DIR"

        echo "Waiting for classifieds to start..."
        sleep 30

        # Populate database
        docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' 2>/dev/null || true

        echo "Classifieds site started at http://localhost:9980"
    fi
}

# Start Homepage
start_homepage() {
    echo ""
    echo "=== Starting Homepage (port 4399) ==="

    # Update index.html with localhost
    perl -pi -e "s|<your-server-hostname>|localhost|g" "$ENV_DIR/webarena-homepage/templates/index.html"

    # Check if homepage is already running
    if pgrep -f "flask run.*4399" > /dev/null; then
        echo "Homepage is already running."
    else
        cd "$ENV_DIR/webarena-homepage"
        nohup flask run --host=0.0.0.0 --port=4399 > /tmp/vwa_homepage.log 2>&1 &
        cd "$PROJECT_DIR"
        echo "Homepage started at http://localhost:4399"
    fi
}

# Main execution
check_required_files

# Ask which sites to start
echo ""
echo "Which sites would you like to start?"
echo "  1) All sites"
echo "  2) Shopping only"
echo "  3) Shopping + Reddit"
echo "  4) Shopping + Reddit + Wikipedia"
echo "  5) Custom selection"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        start_shopping
        start_reddit
        start_wikipedia
        start_classifieds
        start_homepage
        ;;
    2)
        start_shopping
        ;;
    3)
        start_shopping
        start_reddit
        ;;
    4)
        start_shopping
        start_reddit
        start_wikipedia
        ;;
    5)
        read -p "Start shopping? [y/N]: " ans
        [[ $ans =~ ^[Yy]$ ]] && start_shopping
        read -p "Start reddit? [y/N]: " ans
        [[ $ans =~ ^[Yy]$ ]] && start_reddit
        read -p "Start wikipedia? [y/N]: " ans
        [[ $ans =~ ^[Yy]$ ]] && start_wikipedia
        read -p "Start classifieds? [y/N]: " ans
        [[ $ans =~ ^[Yy]$ ]] && start_classifieds
        read -p "Start homepage? [y/N]: " ans
        [[ $ans =~ ^[Yy]$ ]] && start_homepage
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== Docker containers status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "Done! You can now test the sites with:"
echo "  curl -I http://localhost:7770 | head"
echo "  curl -I http://localhost:9999 | head"
echo "  curl -I http://localhost:8888 | head"
echo "  curl -I http://localhost:4399 | head"
echo "  curl -I http://localhost:9980 | head"
