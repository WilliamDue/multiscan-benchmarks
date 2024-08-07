default: bench

.PHONY: clean bench

clean:
	cd common && make clean
	cd data && make clean
	cd filter && make clean
	cd partition && make clean
	cd segreduce && make clean
	cd lexer && make clean
	
bench:
	cd common && make
	cd filter && make
	cd partition && make
	cd segreduce && make
	cd lexer && make