
PKG=visual_behavior
REV=`git rev-parse HEAD | cut -c -7`
UID=`id -u`

clean:

build: clean
	docker build --build-arg PKG=$(PKG) -t $(PKG):latest -f Dockerfile . 

test: build
	docker run -e HOST_ID=$(UID) -v `pwd`/data/:/data visual_behavior:latest /$(PKG)/test.sh

bdist_wheel:
	docker run -e HOST_ID=$(UID) -v `pwd`/data/:/data visual_behavior:latest /$(PKG)/build_bdist_wheel.sh

upload: bdist_wheel
	twine upload `pwd`/data/dist/visual_behavior-*.whl -r local

run-interactive:
	docker run -it $(PKG):latest
