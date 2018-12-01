
PKG=visual_behavior
UID=`id -u`

clean:

	rm -rf fixture

build: clean



	mkdir -p fixtures	
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119102010_421137_c108dc71-ef5e-46ad-8d85-8da0fdaf7d3d.pkl fixtures
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119092559_412629_a3775e3e-e1ca-474a-b413-91cccd6d886f.pkl fixtures
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119150503_416656_2b0893fe-843d-495e-bceb-83b13f2b02dc.pkl fixtures
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119135416_424460_b6daf247-2caf-4f38-9eb1-ab97825923cd.pkl fixtures
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119134201_402329_b75a87d0-8178-4171-a3b2-7cea3ae8e118.pkl fixtures
	cp /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/778113069_stim.pkl fixtures
	docker build --build-arg PKG=$(PKG) -t $(PKG):latest -f Dockerfile . 

test: build
	docker run -t -e HOST_ID=$(UID) -v `pwd`/data/:/data -v /allen/programs/braintv/production/neuralcoding/:/allen/programs/braintv/production/neuralcoding:ro $(PKG):latest /$(PKG)/test.sh

dev:
	docker create -t --rm --name dev  -v /allen/programs/braintv/production/neuralcoding/:/allen/programs/braintv/production/neuralcoding:ro $(PKG):latest /$(PKG)/test.sh
	docker cp test.sh dev:/$(PKG)/test.sh
	find ./tests -name \*.pyc -delete
	rm -rf tests/__pycache__
	docker cp tests dev:/$(PKG)
	docker start -i dev

bdist_wheel:
	docker run -e HOST_ID=$(UID) -v `pwd`/data/:/data visual_behavior:latest /$(PKG)/build_bdist_wheel.sh

upload: bdist_wheel
	twine upload `pwd`/data/dist/visual_behavior-*.whl -r local

run-interactive:
	docker run -it $(PKG):latest
