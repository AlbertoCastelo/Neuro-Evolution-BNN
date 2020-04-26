export IMAGE ?= pyneat

SERVICE := pyneat
SYSTEM_NETWORK := neat

build: create-networks
	@docker build --progress=plain -t $(IMAGE) -f docker/Dockerfile . ;
	#@docker build --no-cache -t $(IMAGE) -f docker/Dockerfile . ;

shell: build
	cd docker && (docker-compose run --service-ports $(SERVICE) /bin/bash) ;

test: build
	cd docker && (docker-compose run --service-ports $(SERVICE) /bin/bash scripts/run_tests.sh) ;

jupyter: build
	cd docker && (docker-compose run --service-ports $(SERVICE) )

create-networks:
	@docker network create $(SYSTEM_NETWORK) > /dev/null 2>&1 || true
