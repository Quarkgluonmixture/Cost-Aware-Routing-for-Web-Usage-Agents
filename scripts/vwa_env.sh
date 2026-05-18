export DATASET=visualwebarena

export CLASSIFIEDS="http://localhost:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

export SHOPPING="http://localhost:7770"
export REDDIT="http://localhost:9999"
export WIKIPEDIA="http://localhost:8888"
export HOMEPAGE="http://localhost:4399"

# B-1574 (/stress A1.24 P1-7-C, 2026-05-18): VWA_*_USER/PASS required by
# p79/utils/auth_refresh.py for auth_required_gate. Set in your shell env
# or override here. Canonical identities per env_config.py:91 + CLAUDE.md:
#   classifieds=blake.sullivan@gmail.com  reddit=MarvelsGrantMan136
#   shopping=emma.lopez@gmail.com         shopping_admin=admin
# Passwords ship with the VWA Docker images (see VWA upstream README).
# export VWA_CLASSIFIEDS_USER="blake.sullivan@gmail.com"
# export VWA_CLASSIFIEDS_PASS="..."
# export VWA_REDDIT_USER="MarvelsGrantMan136"
# export VWA_REDDIT_PASS="..."
# export VWA_SHOPPING_USER="emma.lopez@gmail.com"
# export VWA_SHOPPING_PASS="..."
# export VWA_SHOPPING_ADMIN_USER="admin"
# export VWA_SHOPPING_ADMIN_PASS="..."
