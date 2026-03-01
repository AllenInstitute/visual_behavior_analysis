pip install build --quiet
python -m build --wheel --outdir /data/dist

chown -R $HOST_ID:$HOST_ID /data