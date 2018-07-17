
PKG=visual_behavior
UID=`id -u`

clean:

build: clean
	docker build --build-arg PKG=$(PKG) -t $(PKG):latest -f Dockerfile . 

test: build
	docker run -t -e HOST_ID=$(UID) -v `pwd`/data/:/data visual_behavior:latest /$(PKG)/test.sh

dev:
	docker create -t --rm --name dev $(PKG):latest /$(PKG)/test.sh
	docker cp test.sh dev:/$(PKG)/test.sh
	docker cp tests dev:/$(PKG)
	docker start -i dev

bdist_wheel:
	docker run -e HOST_ID=$(UID) -v `pwd`/data/:/data visual_behavior:latest /$(PKG)/build_bdist_wheel.sh

upload: bdist_wheel
	twine upload `pwd`/data/dist/visual_behavior-*.whl -r local

run-interactive:
	docker run -it $(PKG):latest
