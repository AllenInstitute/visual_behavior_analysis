export DISPLAY=:1.0
Xvfb -br -nolisten tcp -screen 0 1400x900x24 :1.0 &  # Headless mode for running camstim 

coverage run --source visual_behavior -m pytest -v --junitxml=/data/junit.xml --html=/data/test-reports/report.html --self-contained-html

ERRORCODE=$?
coverage report
coverage html --directory=/data/htmlcov

chown -R $HOST_ID:$HOST_ID /data
exit $ERRORCODE
